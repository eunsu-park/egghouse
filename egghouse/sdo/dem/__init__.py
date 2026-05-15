"""
DEM (Differential Emission Measure) analysis module.

Provides tools for computing DEM from multi-wavelength observations,
primarily designed for SDO/AIA EUV imaging data.

Algorithms:
    SITES: Simple Iterative Temperature Emission Solver
           Morgan & Pickering (2019), Solar Physics 294, 135

Modules:
    response: Temperature response functions (aiapy wrapper)
    sites: SITES DEM inversion algorithm
    utils: Map processing and visualization utilities

References:
    - Morgan & Pickering (2019), Solar Physics 294, 135
      DOI: 10.1007/s11207-019-1525-4
    - Hannah & Kontar (2012), A&A 539, A146
"""

from .response import (
    get_temperature_response,
    get_default_temperatures,
    load_ssw_temperature_response,
    HAS_AIAPY,
)
from .sites import (
    dem_sites,
    dem_sites_pixel,
)
from .utils import (
    dem_map,
    compute_dem_errors,
    get_emission_measure,
    get_mean_temperature,
)

__all__ = [
    # Response functions
    "get_temperature_response",
    "get_default_temperatures",
    "load_ssw_temperature_response",
    "HAS_AIAPY",
    # SITES algorithm
    "dem_sites",
    "dem_sites_pixel",
    # Utilities
    "dem_map",
    "compute_dem_errors",
    "get_emission_measure",
    "get_mean_temperature",
]
