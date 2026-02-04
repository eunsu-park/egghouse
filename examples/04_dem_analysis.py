#!/usr/bin/env python
"""
DEM (Differential Emission Measure) Analysis Example
=====================================================

Demonstrates how to compute DEM from multi-wavelength AIA observations
using the SITES algorithm.

Requires: pip install "egghouse[dem]"

Run:
    python examples/04_dem_analysis.py
"""

import numpy as np

# Import DEM module
from egghouse.sdo.dem import (
    get_temperature_response,
    get_default_temperatures,
    dem_sites,
    dem_sites_pixel,
    dem_map,
    get_emission_measure,
    get_mean_temperature,
    HAS_AIAPY,
)


def create_synthetic_observation():
    """Create synthetic AIA-like observation for testing."""
    # Temperature grid
    temps = get_default_temperatures(n_bins=50)
    logt = np.log10(temps)

    # Create a known DEM: isothermal + broad component
    dem_true = (
        1e22 * np.exp(-0.5 * ((logt - 6.2) / 0.15) ** 2)  # Hot component
        + 5e21 * np.exp(-0.5 * ((logt - 5.9) / 0.3) ** 2)  # Warm component
    )

    # Get temperature response (will use fallback if aiapy not installed)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = get_temperature_response(temperatures=temps)

    # Compute synthetic intensities: I = integral(K * DEM * dT)
    dlogt = np.gradient(logt)
    dt = temps * np.log(10) * dlogt
    intensities = np.sum(response * dem_true[:, np.newaxis] * dt[:, np.newaxis], axis=0)

    # Add noise
    noise_level = 0.05
    errors = intensities * noise_level
    intensities_noisy = intensities + np.random.randn(6) * errors

    return {
        "temps": temps,
        "response": response,
        "dem_true": dem_true,
        "intensities": intensities_noisy,
        "errors": errors,
    }


def main():
    print("=" * 60)
    print("egghouse - DEM Analysis Example")
    print("=" * 60)

    # Check aiapy availability
    print(f"\naiapy installed: {HAS_AIAPY}")
    if not HAS_AIAPY:
        print("Note: Using approximate temperature response functions.")
        print("For accurate results, install aiapy: pip install aiapy")

    # 1. Create synthetic data
    print("\n1. Creating Synthetic Observation")
    print("-" * 40)
    data = create_synthetic_observation()
    print(f"   Temperature range: 10^{np.log10(data['temps'][0]):.1f} - "
          f"10^{np.log10(data['temps'][-1]):.1f} K")
    print(f"   Number of temperature bins: {len(data['temps'])}")
    print(f"   AIA wavelengths: [94, 131, 171, 193, 211, 335] Angstrom")
    print(f"   Intensities (DN/s): {data['intensities'].round(1)}")

    # 2. Single-pixel DEM inversion
    print("\n2. Single-Pixel DEM Inversion")
    print("-" * 40)

    dem, info = dem_sites_pixel(
        data["intensities"],
        data["errors"],
        data["response"],
        data["temps"],
        max_iter=100,
        tol=1e-4,
    )

    print(f"   Converged: {info['converged']}")
    print(f"   Iterations: {info['iterations']}")
    print(f"   Chi-squared: {info['chi2']:.2f}")
    print(f"   DEM peak: {dem.max():.2e} cm^-5 K^-1")

    # 3. Compute derived quantities
    print("\n3. Derived Quantities")
    print("-" * 40)

    em = get_emission_measure(dem, data["temps"])
    t_mean = get_mean_temperature(dem, data["temps"])

    print(f"   Total Emission Measure: {em:.2e} cm^-5")
    print(f"   DEM-weighted Mean Temperature: {t_mean/1e6:.2f} MK")
    print(f"   Peak Temperature: {data['temps'][np.argmax(dem)]/1e6:.2f} MK")

    # 4. Compare with true DEM
    print("\n4. Comparison with True DEM")
    print("-" * 40)

    em_true = get_emission_measure(data["dem_true"], data["temps"])
    t_mean_true = get_mean_temperature(data["dem_true"], data["temps"])

    print(f"   True EM: {em_true:.2e}, Recovered: {em:.2e}")
    print(f"   True T_mean: {t_mean_true/1e6:.2f} MK, Recovered: {t_mean/1e6:.2f} MK")

    # 5. DEM map example (small synthetic image)
    print("\n5. DEM Map Processing")
    print("-" * 40)

    # Create small synthetic image cube
    height, width = 16, 16
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response = get_temperature_response(temperatures=data["temps"])

    # Vary DEM spatially
    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    temp_variation = 6.0 + 0.4 * np.sin(2 * np.pi * x_grid / width)

    # Create image cube
    image_cube = np.zeros((height, width, 6), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            dem_local = 1e22 * np.exp(
                -0.5 * ((np.log10(data["temps"]) - temp_variation[i, j]) / 0.2) ** 2
            )
            dlogt = np.gradient(np.log10(data["temps"]))
            dt = data["temps"] * np.log(10) * dlogt
            image_cube[i, j] = np.sum(
                response * dem_local[:, np.newaxis] * dt[:, np.newaxis], axis=0
            )

    error_cube = image_cube * 0.1

    # Run DEM map
    print(f"   Image size: {height}x{width} pixels")
    print("   Processing...")

    dem_cube, map_info = dem_map(
        image_cube,
        error_cube,
        response,
        data["temps"],
        chunk_size=8,
        max_iter=50,
    )

    print(f"   DEM cube shape: {dem_cube.shape}")
    print(f"   Pixels processed: {map_info['n_pixels']}")
    print(f"   Mean iterations: {map_info['mean_iterations']:.1f}")

    # 6. Compute EM and T maps
    print("\n6. EM and Temperature Maps")
    print("-" * 40)

    em_map = get_emission_measure(dem_cube, data["temps"])
    t_map = get_mean_temperature(dem_cube, data["temps"])

    print(f"   EM map range: {em_map.min():.2e} - {em_map.max():.2e} cm^-5")
    print(f"   T_mean map range: {t_map.min()/1e6:.2f} - {t_map.max()/1e6:.2f} MK")

    # 7. Usage pattern
    print("\n7. Typical Usage Pattern")
    print("-" * 40)
    print("""
# Load multi-wavelength AIA data
from egghouse.sdo.dem import (
    get_temperature_response,
    get_default_temperatures,
    dem_map,
    get_emission_measure,
    get_mean_temperature,
)

# Define temperature grid
temps = get_default_temperatures(logt_min=5.5, logt_max=7.5, n_bins=100)

# Get temperature response (requires aiapy for best accuracy)
from datetime import datetime
obs_time = datetime(2024, 1, 15, 12, 0, 0)
response = get_temperature_response(temperatures=temps, time=obs_time)

# Load your AIA images (94, 131, 171, 193, 211, 335 Angstrom)
# image_cube shape: (height, width, 6)
# error_cube shape: (height, width, 6)

# Compute DEM map
dem_cube, info = dem_map(
    image_cube,
    error_cube,
    response,
    temps,
    chunk_size=512,  # Adjust based on memory
    max_iter=100,
)

# Derive physical quantities
em = get_emission_measure(dem_cube, temps)  # Total EM
t_mean = get_mean_temperature(dem_cube, temps)  # DEM-weighted T
""")

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
