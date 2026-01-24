"""
SDO (Solar Dynamics Observatory) data utilities.

Provides functions for processing and visualizing SDO/AIA and SDO/HMI data.

Modules:
    aia: AIA intensity scaling and calibration utilities.
    hmi: HMI magnetogram processing utilities.
    core: Common utilities for SDO data (FITS parsing, validation).
"""

from .aia import aia_intscale, AIA_CALIBRATION, get_aia_calibration
from .hmi import hmi_intscale, hmi_field_strength
from .core import (
    parse_fits_header,
    validate_sdo_image,
    get_solar_disk_params,
    HAS_ASTROPY,
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
]
