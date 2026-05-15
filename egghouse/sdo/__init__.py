"""
SDO (Solar Dynamics Observatory) data utilities.

Provides functions for processing and visualizing SDO/AIA and SDO/HMI data.

Modules:
    aia: AIA intensity scaling and calibration utilities.
    hmi: HMI magnetogram processing utilities.
    core: Common utilities for SDO data (FITS parsing, validation).
    stacking: Solar rotation-corrected image stacking for HMI.
    level15: Level 1.0 to Level 1.5 preprocessing.
    quality: QUALITY keyword interpretation utilities.
    dem: DEM (Differential Emission Measure) analysis using SITES algorithm.
"""

from .aia import aia_intscale, AIA_CALIBRATION, get_aia_calibration
from .hmi import hmi_intscale, hmi_field_strength
from .core import (
    parse_fits_header,
    validate_sdo_image,
    get_solar_disk_params,
    HAS_ASTROPY,
)
from .level15 import (
    to_level15,
    batch_to_level15,
    get_level_info,
    AIA_PLATE_SCALE,
    HMI_PLATE_SCALE,
    SDO_IMAGE_SIZE,
)
from .stacking import (
    # Classes
    Stacking,
    StreamingStackAccumulator,
    # Functions
    stack_with_rotation_correction,
    solar_rotation_shift,
    snodgrass_rotation_rate,
    detect_cadence_from_maps,
    cross_correlate_shift,
    # Constants
    SOLAR_ROTATION_PERIOD,
    SNODGRASS_A,
    SNODGRASS_B,
    SNODGRASS_C,
    HMI_CADENCE_45S,
    HMI_CADENCE_720S,
    # Type aliases
    StackingMethod,
    ProgressCallback,
    # Flags
    HAS_SUNPY,
)
from .quality import (
    decode_quality,
    format_quality,
    is_quality_ok,
    get_quality_summary,
    print_all_quality_bits,
    AIA_QUALITY_BITS,
    HMI_QUALITY_BITS,
    QUALLEV0_BITS,
)
from .jsoc import (
    jsoc_export,
    aia_euv_query,
    cached_correction_table,
    cached_pointing_table,
    AIA_LEV1_EUV_SERIES,
)
from .dem import (
    # Response functions
    get_temperature_response,
    get_default_temperatures,
    load_ssw_temperature_response,
    HAS_AIAPY,
    # SITES algorithm
    dem_sites,
    dem_sites_pixel,
    # Utilities
    dem_map,
    compute_dem_errors,
    get_emission_measure,
    get_mean_temperature,
)

__all__ = [
    # AIA
    'aia_intscale',
    'AIA_CALIBRATION',
    'get_aia_calibration',
    # HMI
    'hmi_intscale',
    'hmi_field_strength',
    # Core
    'parse_fits_header',
    'validate_sdo_image',
    'get_solar_disk_params',
    'HAS_ASTROPY',
    # Level 1.5
    'to_level15',
    'batch_to_level15',
    'get_level_info',
    'AIA_PLATE_SCALE',
    'HMI_PLATE_SCALE',
    'SDO_IMAGE_SIZE',
    # Stacking - Classes
    'Stacking',
    'StreamingStackAccumulator',
    # Stacking - Functions
    'stack_with_rotation_correction',
    'solar_rotation_shift',
    'snodgrass_rotation_rate',
    'detect_cadence_from_maps',
    'cross_correlate_shift',
    # Stacking - Constants
    'SOLAR_ROTATION_PERIOD',
    'SNODGRASS_A',
    'SNODGRASS_B',
    'SNODGRASS_C',
    'HMI_CADENCE_45S',
    'HMI_CADENCE_720S',
    # Stacking - Type aliases
    'StackingMethod',
    'ProgressCallback',
    # Stacking - Flags
    'HAS_SUNPY',
    # Quality - Functions
    'decode_quality',
    'format_quality',
    'is_quality_ok',
    'get_quality_summary',
    'print_all_quality_bits',
    # Quality - Constants
    'AIA_QUALITY_BITS',
    'HMI_QUALITY_BITS',
    'QUALLEV0_BITS',
    # JSOC export
    'jsoc_export',
    'aia_euv_query',
    'cached_correction_table',
    'cached_pointing_table',
    'AIA_LEV1_EUV_SERIES',
    # DEM - Response functions
    'get_temperature_response',
    'get_default_temperatures',
    'load_ssw_temperature_response',
    'HAS_AIAPY',
    # DEM - SITES algorithm
    'dem_sites',
    'dem_sites_pixel',
    # DEM - Utilities
    'dem_map',
    'compute_dem_errors',
    'get_emission_measure',
    'get_mean_temperature',
]
