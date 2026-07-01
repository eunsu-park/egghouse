# egghouse.image User Guide

General-purpose image processing utilities. Built on scipy.ndimage, preserving dtype during processing.

---

## Module Structure

```
egghouse/image/
├── __init__.py   # exports all functions
├── core.py       # basic transforms (resize, rotate, bytescale)
├── masking.py    # masking (circle_mask, annulus_mask)
├── spatial.py    # spatial transforms (pad, crop_or_pad, flip, roll)
├── filters.py    # filtering (gaussian, median, laplacian, sobel, unsharp)
├── stats.py      # statistics/analysis (normalize, histogram_eq, percentile_scale, find_center)
├── metrics.py    # image quality metrics (psnr, ssim, ms_ssim, weak_signal_contrast) [v0.9+]
├── transforms.py # composable numpy transforms (compose, percentile_clip, ...) [v0.9+]
└── noise.py      # robust noise scale (mad, robust_sigma) [v0.9+]
```

---

## Function List

### Core (basic transforms)
| Function | Description | Alias |
|------|------|-------|
| `resize_image` | Resize an image | `resize` |
| `rotate_image` | Rotate an image | `rotate` |
| `bytescale_image` | Scale to uint8 | `bytescale` |

### Masking
| Function | Description |
|------|------|
| `circle_mask` | Create a circular mask |
| `annulus_mask` | Create an annular (ring) mask |

### Spatial (spatial transforms)
| Function | Description | Alias |
|------|------|-------|
| `pad_image` | Add padding | `pad` |
| `crop_or_pad` | Adjust to an exact size | - |
| `flip_image` | Flip an image | - |
| `roll_image` | Cyclic shift | - |
| `bin_ndarray` | Block-wise n-dimensional down-binning (v0.5+) | - |

### Filters (filtering)
| Function | Description |
|------|------|
| `gaussian_smooth` | Gaussian smoothing |
| `median_denoise` | Median noise removal |
| `laplacian_edge` | Laplacian edge detection |
| `sobel_edge` | Sobel edge detection |
| `unsharp_mask` | Unsharp mask sharpening |

### Stats (statistics/analysis)
| Function | Description |
|------|------|
| `normalize_image` | z-score normalization |
| `get_image_stats` | Compute image statistics |
| `histogram_equalization` | Histogram equalization |
| `percentile_scale` | percentile-based scaling |
| `find_disk_center` | Find the disk center |
| `adaptive_threshold` | Adaptive thresholding |

### Metrics (image quality metrics, v0.9+)
| Function | Description |
|------|------|
| `psnr` | Peak SNR (dB); `+inf` for identical images |
| `ssim` | Single-scale SSIM (Wang 2004) |
| `ms_ssim` | Multi-scale SSIM (Wang 2003, 5-scale) |
| `weak_signal_contrast` | Sobel gradient correlation (weak-edge preservation, placeholder) |

```python
from egghouse.image import psnr, ssim, ms_ssim
score = psnr(denoised, reference, data_range=4.0)
```

### Transforms (composable transforms, v0.9+)
Chain with `compose([...])`. Each function maps a numpy array to a numpy array.

| Function | Description |
|------|------|
| `compose` | Chain transforms left-to-right |
| `to_float32` | Native float32 cast (no value change) |
| `nan_to_value` | Replace NaN/Inf with a given value |
| `percentile_clip` | Clip to a per-frame percentile range |
| `normalize_minmax` | Per-frame [0,1] scaling |
| `normalize_log1p` | `log1p(scale*(x-min))` dynamic-range compression |
| `circular_mask` | Fill inside/outside of a center circle |

```python
from egghouse.image import transforms as T
pipe = T.compose([T.to_float32, T.percentile_clip(0.5, 99.5), T.normalize_minmax()])
out = pipe(frame)
```

### Noise (robust noise scale, v0.9+)
| Function | Description |
|------|------|
| `mad` | Median absolute deviation about the median |
| `robust_sigma` | Robust σ estimate `1.4826 * MAD` (robust to outliers) |

```python
from egghouse.image import robust_sigma
sigma = robust_sigma(frame)   # not pulled by hot pixels/transients
```

---

## Core Functions

### resize_image

Resize an image. Preserves dtype.

```python
from egghouse.image import resize_image

# Basic usage (bilinear interpolation)
resized = resize_image(image, (512, 512))

# Choose the interpolation method
resized = resize_image(image, (256, 256), order=0)  # nearest
resized = resize_image(image, (256, 256), order=1)  # bilinear (default)
resized = resize_image(image, (256, 256), order=3)  # bicubic

# 3D image (H, W, C) support
rgb_resized = resize_image(rgb_image, (256, 256))
```

**order values:**
- 0: nearest-neighbor (fastest, staircase artifacts)
- 1: bilinear (default, balanced)
- 2: bi-quadratic
- 3: bi-cubic (smoothest, slow)

---

### rotate_image

Rotate an image. Positive angles are counter-clockwise.

```python
from egghouse.image import rotate_image

# Basic rotation (keeps original size)
rotated = rotate_image(image, angle=45)

# Expand the canvas so the whole image is visible
rotated = rotate_image(image, angle=45, reshape=True)

# Specify the fill value for empty areas
rotated = rotate_image(image, angle=30, cval=np.nan)
```

**Parameters:**
- `angle`: rotation angle (in degrees)
- `reshape`: if True, resizes the output so the whole image is visible
- `cval`: fill value for out-of-bounds areas

---

### bytescale_image

Scale data to the uint8 range [0, 255]. For visualization.

```python
from egghouse.image import bytescale_image

# Automatic range detection
display = bytescale_image(data)

# Specify the range
display = bytescale_image(data, imin=0, imax=5000)

# percentile-based contrast stretch
p1, p99 = np.percentile(data, [1, 99])
display = bytescale_image(data, imin=p1, imax=p99)

# Change the output range
display = bytescale_image(data, omin=10, omax=245)  # reserve margins
```

---

## Masking Functions

### circle_mask

Create a circular boolean mask. Useful for masking the solar disk.

```python
from egghouse.image import circle_mask

# Solar disk mask (4096x4096, radius 1600 pixels)
disk_mask = circle_mask(4096, radius=1600)

# Extract only the disk interior
masked = np.where(disk_mask, image, 0)

# Mask outside the disk (for corona analysis)
corona_mask = circle_mask(4096, radius=1600, mask_type='outer')

# Rectangular image, with a specified center
mask = circle_mask((512, 1024), radius=200, center=(256, 600))
```

**mask_type:**
- `'inner'`: interior of the circle is True (default)
- `'outer'`: exterior of the circle is True

---

### annulus_mask

Create an annular (ring) mask. For analyzing a specific radius range.

```python
from egghouse.image import annulus_mask

# Region between 1.0 and 1.3 solar radii
solar_radius = 1600
corona_ring = annulus_mask(4096,
                           inner_radius=solar_radius,
                           outer_radius=solar_radius * 1.3)

# Apply the mask
corona_data = image[corona_ring]
mean_intensity = corona_data.mean()
```

---

## Spatial Functions

### pad_image

Add padding to an image.

```python
from egghouse.image import pad_image

# Center-aligned padding (default)
padded = pad_image(image, (1024, 1024), pad_value=0)

# Top-left aligned
padded = pad_image(image, (1024, 1024), center=False)

# Pad with NaN (off-disk areas)
padded = pad_image(image, (5000, 5000), pad_value=np.nan)
```

---

### crop_or_pad

Automatically crop or pad to fit a size.

```python
from egghouse.image import crop_or_pad

# Fit images of varying sizes to the same size
img1 = np.random.rand(400, 600)   # small image → pad
img2 = np.random.rand(800, 500)   # large image → crop

normalized1 = crop_or_pad(img1, (512, 512))
normalized2 = crop_or_pad(img2, (512, 512))
# both are (512, 512)
```

---

### flip_image

Flip an image.

```python
from egghouse.image import flip_image

# Vertical flip (default)
flipped = flip_image(image, axis='vertical')

# Horizontal flip
flipped = flip_image(image, axis='horizontal')

# 180-degree rotation (flip both)
flipped = flip_image(image, axis='both')
```

**axis:**
- `'vertical'`: vertical flip (default)
- `'horizontal'`: horizontal flip
- `'both'`: both (180-degree rotation)

---

### roll_image

Cyclic shift. Pixels crossing a boundary reappear on the opposite side.

```python
from egghouse.image import roll_image

# Shift down 10 pixels, right 5 pixels
rolled = roll_image(image, shift_y=10, shift_x=5)

# Use for image alignment
for i, img in enumerate(images):
    aligned = roll_image(img, shift_y=0, shift_x=shifts[i])
```

---

### bin_ndarray

`bin_ndarray(array, new_shape, operation='sum')` (v0.5+)

Block-wise n-dimensional down-binning. Reshapes the array into
`(n0, b0, n1, b1, ...)` form and then reduces each block by summation
(`'sum'`) or averaging (`'mean'`) (no overlap). `new_shape` must have the
same number of dimensions as the input, and each output dimension must
divide the corresponding input dimension exactly.

Parameters:

- `array`: input n-dimensional array.
- `new_shape`: target shape (tuple). Must have the same length as
  `array.ndim`, and each element must divide `array.shape[i]` exactly.
- `operation`: `'sum'` (default) or `'mean'`. The block reduction method.

Raises `ValueError` if `operation` is neither `'sum'` nor `'mean'`, if the
number of dimensions does not match, or if any dimension does not divide
evenly.

```python
import numpy as np
from egghouse.image import bin_ndarray

# Down-bin a 4096x4096 solar image to 1024x1024 (4x4 blocks)
solar = np.random.rand(4096, 4096)

# Block sum: preserves additive quantities such as photon counts
binned_sum = bin_ndarray(solar, (1024, 1024), operation='sum')
print(binned_sum.shape)  # (1024, 1024)

# Block mean: keeps intensity/brightness scale
binned_mean = bin_ndarray(solar, (1024, 1024), operation='mean')
print(binned_mean.shape)  # (1024, 1024)
```

---

## Filters Functions

### gaussian_smooth

Reduce noise with a Gaussian filter. Natural smoothing.

```python
from egghouse.image import gaussian_smooth

# Basic smoothing (sigma=1.0)
smoothed = gaussian_smooth(image, sigma=1.5)

# Different sigma per axis
smoothed = gaussian_smooth(image, sigma=(2.0, 1.0))

# Do not preserve dtype
smoothed = gaussian_smooth(image, sigma=1.0, preserve_range=False)
```

---

### median_denoise

Remove noise with a median filter. Excellent at preserving edges. Effective for salt-and-pepper noise and cosmic ray removal.

```python
from egghouse.image import median_denoise

# Default (3x3 window)
denoised = median_denoise(image, size=3)

# For stronger noise
denoised = median_denoise(noisy_image, size=5)

# Non-square window
denoised = median_denoise(image, size=(3, 5))
```

---

### laplacian_edge

Laplacian edge detection. Detects abrupt changes via the second derivative.

```python
from egghouse.image import laplacian_edge

# Basic edge detection
edges = laplacian_edge(image)

# Apply after Gaussian pre-processing (LoG)
smoothed = gaussian_smooth(image, sigma=1.0)
edges = laplacian_edge(smoothed)
```

---

### sobel_edge

Sobel edge detection. Based on the first derivative (gradient).

```python
from egghouse.image import sobel_edge

# Gradient magnitude (all edges)
edges = sobel_edge(image)

# Vertical edges only (gradient in the y direction)
edges_y = sobel_edge(image, axis=0)

# Horizontal edges only (gradient in the x direction)
edges_x = sobel_edge(image, axis=1)
```

---

### unsharp_mask

Unsharp mask sharpening. Subtracts a blurred image to emphasize edges.

```python
from egghouse.image import unsharp_mask

# Basic sharpening
sharp = unsharp_mask(image, sigma=1.0, amount=1.0)

# Stronger sharpening
sharp = unsharp_mask(image, sigma=2.0, amount=2.0)
```

**Parameters:**
- `sigma`: blur strength. Higher values emphasize wider edges
- `amount`: sharpening strength. Values above 1.0 are stronger

---

## Stats Functions

### normalize_image

z-score normalization. Transforms to mean 0, standard deviation 1.

```python
from egghouse.image import normalize_image

# Automatic computation
normalized = normalize_image(image)
# mean ≈ 0, std ≈ 1

# Use pre-computed statistics (based on the training set)
normalized = normalize_image(image, mean=127.5, std=64.0)
```

---

### get_image_stats

Compute image statistics. Supports masks.

```python
from egghouse.image import get_image_stats

# Whole-image statistics
stats = get_image_stats(image)
print(f"Mean: {stats['mean']:.2f}")
print(f"Std: {stats['std']:.2f}")
print(f"Min: {stats['min']}, Max: {stats['max']}")
print(f"p1={stats['p1']}, p99={stats['p99']}")

# Solar disk interior only
disk_mask = circle_mask(4096, radius=1600)
stats = get_image_stats(image, mask=disk_mask)

# Custom percentiles
stats = get_image_stats(image, percentiles=(5, 50, 95))
```

**Returns:**
- `mean`, `std`, `min`, `max`, `median`, `count`
- `p1`, `p5`, `p25`, `p50`, `p75`, `p95`, `p99` (or custom)

---

### histogram_equalization

Histogram equalization. Makes a narrow intensity distribution uniform.

```python
from egghouse.image import histogram_equalization

# Improve a low-contrast image
enhanced = histogram_equalization(image)

# Comparison visualization
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(image, cmap='gray')
ax1.set_title('Original')
ax2.imshow(enhanced, cmap='gray')
ax2.set_title('Equalized')
```

---

### percentile_scale

percentile-based scaling. Robust to outliers.

```python
from egghouse.image import percentile_scale

# Default (1%, 99%)
scaled = percentile_scale(image)

# More aggressive clipping
scaled = percentile_scale(image, low_percentile=5, high_percentile=95)

# Custom output range
scaled = percentile_scale(image, omin=10, omax=245)
```

Similar to bytescale_image, but automatically uses a percentile-based range.

---

### find_disk_center

Find the center coordinates of a bright disk (e.g. the Sun).

```python
from egghouse.image import find_disk_center

# Find the solar disk center
cy, cx = find_disk_center(aia_image)
print(f"Center: ({cy:.1f}, {cx:.1f})")

# Custom threshold
cy, cx = find_disk_center(image, threshold=100)

# Geometric center (ignore intensity)
cy, cx = find_disk_center(image, method='geometric')

# Create a mask from the found center
mask = circle_mask(image.shape, radius=1600, center=(cy, cx))
```

**method:**
- `'centroid'`: brightness-weighted center (default)
- `'geometric'`: geometric center

---

### adaptive_threshold

Adaptive thresholding. Handles non-uniform illumination.

```python
from egghouse.image import adaptive_threshold

# Basic adaptive threshold
binary = adaptive_threshold(image)

# More sensitive (more foreground)
binary = adaptive_threshold(image, offset=-5)

# Smaller blocks (preserve detail)
binary = adaptive_threshold(image, block_size=15)
```

**Parameters:**
- `block_size`: window size for the local mean computation (odd)
- `offset`: value subtracted from the mean. Positive values reduce foreground

---

## Workflow Examples

### Solar Image Processing

```python
from astropy.io import fits  # egghouse.io was removed in v0.6.0
from egghouse.image import (
    resize_image, circle_mask, bytescale_image, crop_or_pad,
    gaussian_smooth, find_disk_center, get_image_stats
)

# 1. Read the FITS file
data, header = fits.getdata('aia_171.fits', header=True)

# 2. Normalize the size
data = crop_or_pad(data, (4096, 4096))

# 3. Find the disk center and apply a mask
cy, cx = find_disk_center(data)
disk_mask = circle_mask(4096, radius=1600, center=(cy, cx))
data = np.where(disk_mask, data, 0)

# 4. Check statistics
stats = get_image_stats(data, mask=disk_mask)
print(f"Disk mean: {stats['mean']:.1f}")

# 5. Denoise
data = gaussian_smooth(data, sigma=1.0)

# 6. Resize (for ML input)
resized = resize_image(data, (512, 512), order=1)

# 7. Scale for visualization
display = bytescale_image(resized, imin=0, imax=5000)
```

### Edge Detection

```python
from egghouse.image import gaussian_smooth, sobel_edge, laplacian_edge

# Detect edges after denoising
smoothed = gaussian_smooth(image, sigma=1.5)

# Sobel: gradient magnitude
edges_sobel = sobel_edge(smoothed)

# Laplacian: second derivative
edges_laplacian = laplacian_edge(smoothed)
```

---

## dtype Preservation

All functions preserve the original dtype with `preserve_range=True` (default):

```python
import numpy as np

# uint16 input
img_uint16 = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)

# Stays uint16 after processing
resized = resize_image(img_uint16, (50, 50))
print(resized.dtype)  # uint16

smoothed = gaussian_smooth(img_uint16, sigma=1.0)
print(smoothed.dtype)  # uint16
```

Setting `preserve_range=False` returns float64:

```python
resized = resize_image(img_uint16, (50, 50), preserve_range=False)
print(resized.dtype)  # float64
```

---

## Dependencies

- numpy
- scipy (ndimage)

Install: `pip install numpy scipy`
