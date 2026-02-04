#!/usr/bin/env python
"""
Basic Image Processing Example
==============================

Demonstrates basic image processing with egghouse in 5 minutes.

Run:
    python examples/01_basic_image.py
"""

import numpy as np

from egghouse.image import (
    resize_image,
    rotate_image,
    bytescale_image,
    circle_mask,
    gaussian_smooth,
    percentile_scale,
    get_image_stats,
)


def main():
    print("=" * 60)
    print("egghouse - Basic Image Processing Example")
    print("=" * 60)

    # 1. Create a sample image (simulated solar disk)
    print("\n1. Creating sample solar disk image...")
    size = 256
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    center = size // 2
    radius = size // 3

    disk = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) < radius
    image = np.zeros((size, size), dtype=np.float32)
    image[disk] = 1000 + 200 * np.sin(xx[disk] / 10) * np.cos(yy[disk] / 10)
    image += np.random.randn(size, size).astype(np.float32) * 50

    print(f"   Shape: {image.shape}, dtype: {image.dtype}")
    print(f"   Range: [{image.min():.1f}, {image.max():.1f}]")

    # 2. Resize
    print("\n2. Resizing image...")
    small = resize_image(image, (64, 64))
    large = resize_image(image, (512, 512))
    print(f"   Original: {image.shape}")
    print(f"   Small:    {small.shape}")
    print(f"   Large:    {large.shape}")

    # 3. Rotate
    print("\n3. Rotating image...")
    rotated = rotate_image(image, 45)
    print(f"   Rotated 45 degrees: {rotated.shape}")

    # 4. Create circular mask
    print("\n4. Creating circular mask...")
    mask = circle_mask(size, radius=80)
    masked_image = np.where(mask, image, 0)
    print(f"   Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"   Pixels inside mask: {mask.sum()}")

    # 5. Apply Gaussian smoothing
    print("\n5. Applying Gaussian smoothing...")
    smoothed = gaussian_smooth(image, sigma=2.0)
    print(f"   Smoothed shape: {smoothed.shape}")

    # 6. Get image statistics
    print("\n6. Computing image statistics...")
    stats = get_image_stats(image, mask=mask)
    print(f"   Mean:   {stats['mean']:.2f}")
    print(f"   Std:    {stats['std']:.2f}")
    print(f"   Min:    {stats['min']:.2f}")
    print(f"   Max:    {stats['max']:.2f}")
    print(f"   Median: {stats['median']:.2f}")

    # 7. Scale to uint8
    print("\n7. Scaling to uint8...")
    scaled = bytescale_image(image)
    print(f"   Scaled dtype: {scaled.dtype}")
    print(f"   Scaled range: [{scaled.min()}, {scaled.max()}]")

    # 8. Percentile scaling (robust to outliers)
    print("\n8. Percentile scaling...")
    percentile = percentile_scale(image, low_percentile=1, high_percentile=99)
    print(f"   Percentile scaled: {percentile.dtype}")

    print("\n" + "=" * 60)
    print("Done! All basic operations completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
