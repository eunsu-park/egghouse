"""
Level 1.5 preprocessing for SDO AIA and HMI data.

Converts Level 1.0 FITS files to Level 1.5 format:
- North-up alignment (CROTA2 = 0)
- Centered sun (CRPIX at image center)
- Standardized plate scale
- Fixed 4096x4096 output size
"""

import os
import warnings
from typing import Callable, List, Literal, Optional, Union

import numpy as np

# Optional dependencies
try:
    from sunpy.map import Map
    import astropy.units as u
    HAS_SUNPY = True
except ImportError:
    HAS_SUNPY = False


# =============================================================================
# Constants
# =============================================================================

# Standard plate scales (arcsec/pixel)
AIA_PLATE_SCALE = 0.6
HMI_PLATE_SCALE = 0.5

# Standard SDO image size
SDO_IMAGE_SIZE = 4096

# Type alias for progress callback
ProgressCallback = Callable[[int, int, str], None]


# =============================================================================
# Main Functions
# =============================================================================

def to_level15(
    fits_file: str,
    instrument: Literal['AIA', 'HMI'] = 'AIA',
    target_plate_scale: Optional[float] = None,
    target_size: int = SDO_IMAGE_SIZE,
    order: int = 3,
    missing: float = np.nan
) -> "Map":
    """
    Convert Level 1.0 FITS file to Level 1.5.

    Level 1.5 processing includes:
    1. Rotation to align solar north with image up (CROTA2 → 0)
    2. Resampling to standard plate scale (0.6 arcsec/px for AIA, 0.5 for HMI)
    3. Resizing to 4096x4096 fixed output

    Args:
        fits_file: Path to Level 1.0 FITS file.
        instrument: 'AIA' or 'HMI'. Determines default plate scale.
        target_plate_scale: Override default plate scale (arcsec/pixel).
            If None, uses 0.6 for AIA or 0.5 for HMI.
        target_size: Output image size in pixels. Defaults to 4096.
        order: Interpolation order (0-5). Defaults to 3 (bicubic).
        missing: Fill value for missing/rotated pixels. Defaults to NaN.

    Returns:
        sunpy.map.Map object with Level 1.5 properties:
            - CROTA2 = 0 (north-up)
            - CRPIX1/2 at image center
            - CDELT1/2 = target_plate_scale
            - LVL_NUM = 1.5

    Raises:
        ImportError: If sunpy is not installed.
        FileNotFoundError: If fits_file does not exist.

    Example:
        >>> from egghouse.sdo import to_level15
        >>> # Convert AIA 171 image
        >>> m = to_level15('aia_171.fits', instrument='AIA')
        >>> assert m.meta['CROTA2'] == 0.0
        >>> assert m.data.shape == (4096, 4096)

        >>> # Convert HMI magnetogram
        >>> m_hmi = to_level15('hmi_m.fits', instrument='HMI')
    """
    if not HAS_SUNPY:
        raise ImportError(
            "sunpy is required for Level 1.5 processing. "
            "Install with: pip install sunpy"
        )

    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    # Determine target plate scale
    if target_plate_scale is None:
        target_plate_scale = AIA_PLATE_SCALE if instrument == 'AIA' else HMI_PLATE_SCALE

    # Load FITS file
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        m = Map(fits_file)

    # Get current rotation angle
    crota2 = m.meta.get('CROTA2', 0.0)

    # Step 1: Rotate to remove CROTA2 (align north up)
    # Note: We rotate by -CROTA2 to bring north to the top
    if abs(crota2) > 0.01:  # Only rotate if there's significant rotation
        m_rotated = m.rotate(
            angle=-crota2 * u.deg,
            order=order,
            missing=missing,
            recenter=True
        )
    else:
        m_rotated = m

    # Step 2: Resample to target size
    m_resampled = m_rotated.resample([target_size, target_size] * u.pix)

    # Step 3: Update metadata
    # Calculate new plate scale based on resampling
    original_fov_x = abs(m.meta.get('CDELT1', 0.6)) * m.data.shape[1]
    original_fov_y = abs(m.meta.get('CDELT2', 0.6)) * m.data.shape[0]
    new_cdelt1 = original_fov_x / target_size
    new_cdelt2 = original_fov_y / target_size

    # Update header values
    m_resampled.meta['CROTA2'] = 0.0
    m_resampled.meta['CDELT1'] = new_cdelt1
    m_resampled.meta['CDELT2'] = new_cdelt2
    m_resampled.meta['CRPIX1'] = (target_size + 1) / 2.0
    m_resampled.meta['CRPIX2'] = (target_size + 1) / 2.0
    m_resampled.meta['NAXIS1'] = target_size
    m_resampled.meta['NAXIS2'] = target_size
    m_resampled.meta['LVL_NUM'] = 1.5

    return m_resampled


def batch_to_level15(
    fits_files: List[str],
    output_dir: str,
    instrument: Literal['AIA', 'HMI'] = 'AIA',
    overwrite: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    **kwargs
) -> List[str]:
    """
    Batch convert multiple FITS files to Level 1.5.

    Args:
        fits_files: List of paths to Level 1.0 FITS files.
        output_dir: Directory to save Level 1.5 FITS files.
        instrument: 'AIA' or 'HMI'. Determines default plate scale.
        overwrite: If True, overwrite existing output files.
        progress_callback: Optional callback(current, total, message) for progress.
        **kwargs: Additional arguments passed to to_level15().

    Returns:
        List of output file paths.

    Raises:
        ImportError: If sunpy is not installed.
        FileNotFoundError: If output_dir does not exist.

    Example:
        >>> from egghouse.sdo import batch_to_level15
        >>> files = ['aia_001.fits', 'aia_002.fits', 'aia_003.fits']
        >>> output_files = batch_to_level15(files, '/output/', instrument='AIA')
    """
    if not HAS_SUNPY:
        raise ImportError(
            "sunpy is required for Level 1.5 processing. "
            "Install with: pip install sunpy"
        )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_files = []
    n_files = len(fits_files)

    for idx, fits_file in enumerate(fits_files):
        if progress_callback:
            progress_callback(idx + 1, n_files, f"Processing {os.path.basename(fits_file)}")

        # Generate output filename
        basename = os.path.basename(fits_file)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name}_level15{ext}")

        # Skip if exists and not overwriting
        if os.path.exists(output_path) and not overwrite:
            output_files.append(output_path)
            continue

        try:
            # Convert to Level 1.5
            m_level15 = to_level15(fits_file, instrument=instrument, **kwargs)

            # Save to FITS
            m_level15.save(output_path, overwrite=overwrite)
            output_files.append(output_path)

        except Exception as e:
            warnings.warn(f"Failed to process {fits_file}: {e}")
            continue

    return output_files


def get_level_info(fits_file: str) -> dict:
    """
    Get processing level information from a FITS file.

    Args:
        fits_file: Path to FITS file.

    Returns:
        Dictionary containing:
            - level: Processing level (1.0, 1.5, etc.)
            - crota2: Current rotation angle
            - cdelt1/cdelt2: Current plate scale
            - crpix1/crpix2: Current reference pixel
            - is_level15: Boolean indicating if already Level 1.5

    Example:
        >>> info = get_level_info('aia_171.fits')
        >>> if not info['is_level15']:
        ...     m = to_level15('aia_171.fits')
    """
    if not HAS_SUNPY:
        raise ImportError(
            "sunpy is required. Install with: pip install sunpy"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        m = Map(fits_file)

    meta = m.meta
    crota2 = meta.get('CROTA2', 0.0)
    cdelt1 = meta.get('CDELT1', 0.0)
    cdelt2 = meta.get('CDELT2', 0.0)
    crpix1 = meta.get('CRPIX1', 0.0)
    crpix2 = meta.get('CRPIX2', 0.0)
    lvl_num = meta.get('LVL_NUM', 1.0)
    naxis1 = meta.get('NAXIS1', 0)
    naxis2 = meta.get('NAXIS2', 0)

    # Check if Level 1.5 criteria are met
    is_centered = (
        abs(crpix1 - (naxis1 + 1) / 2.0) < 1.0 and
        abs(crpix2 - (naxis2 + 1) / 2.0) < 1.0
    )
    is_north_up = abs(crota2) < 0.01
    is_level15 = is_north_up and is_centered and lvl_num >= 1.5

    return {
        'level': lvl_num,
        'crota2': crota2,
        'cdelt1': cdelt1,
        'cdelt2': cdelt2,
        'crpix1': crpix1,
        'crpix2': crpix2,
        'naxis1': naxis1,
        'naxis2': naxis2,
        'is_north_up': is_north_up,
        'is_centered': is_centered,
        'is_level15': is_level15
    }
