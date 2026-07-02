"""FITS observation-datetime parsers for SDO and heliophysics imagery.

Promoted verbatim from ``solaris-data/core/parse.py`` (the FITS-datetime
cluster) so every SOLARIS sub-project shares one tested implementation.
The pure string/filename parsers depend only on the standard library;
:func:`parse_fits_datetime` additionally reads a FITS header via astropy
(lazy-imported) and falls back to filename parsing when the header is
absent or unreadable.

Parsing logic is byte-identical to the original; only the astropy import
was made lazy (``HAS_ASTROPY``) to match egghouse conventions.
"""
from __future__ import annotations

import re
from datetime import datetime

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

__all__ = ["parse_fits_datetime"]


def _parse_tai_datetime(date_str: str) -> datetime | None:
    """Parse TAI format datetime string.

    Handles SDO TAI format: YYYY.MM.DD_HH:MM:SS_TAI
    Note: TAI to UTC conversion is not performed (difference ~35s).

    Args:
        date_str: TAI format datetime string.

    Returns:
        Datetime object or None if parsing failed.
    """
    if not date_str or '_TAI' not in date_str:
        return None

    # Remove _TAI suffix
    date_str = date_str.replace('_TAI', '')

    # Try YYYY.MM.DD_HH:MM:SS format
    try:
        return datetime.strptime(date_str, '%Y.%m.%d_%H:%M:%S')
    except ValueError:
        pass

    # Try YYYY.MM.DD_HH:MM:SS.fff format (with microseconds)
    try:
        return datetime.strptime(date_str, '%Y.%m.%d_%H:%M:%S.%f')
    except ValueError:
        pass

    return None


def _parse_datetime_string(date_str: str) -> datetime | None:
    """Parse datetime from various string formats.

    Args:
        date_str: Date string to parse.

    Returns:
        Datetime object or None if parsing failed.
    """
    if not date_str:
        return None

    date_str = str(date_str).strip()

    # TAI format (SDO HMI/AIA): YYYY.MM.DD_HH:MM:SS_TAI
    if '_TAI' in date_str:
        result = _parse_tai_datetime(date_str)
        if result:
            return result

    # ISO formats with 'T' separator
    if 'T' in date_str:
        # Remove trailing 'Z' or timezone info
        date_str = date_str.rstrip('Z')

        for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                    '%Y/%m/%dT%H:%M:%S.%f', '%Y/%m/%dT%H:%M:%S']:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

    # Date + time with space separator
    for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S',
                '%d/%m/%y %H:%M:%S.%f', '%d/%m/%y %H:%M:%S',
                '%d/%m/%Y %H:%M:%S.%f', '%d/%m/%Y %H:%M:%S']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Date-only formats
    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%y', '%d/%m/%Y']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def _parse_datetime_from_filename(file_path: str) -> datetime | None:
    """Extract datetime from filename patterns.

    Supports:
    - AIA: aia.lev1_euv_12s.2010-09-01T000008Z.193.image_lev1.fits
    - HMI: hmi.m_45s.20100901_000000_TAI.2.magnetogram.fits

    Args:
        file_path: Path to the file.

    Returns:
        Datetime object or None if parsing failed.
    """
    import os
    filename = os.path.basename(file_path)

    # AIA pattern: YYYY-MM-DDTHHMMSSZ
    aia_match = re.search(r'(\d{4}-\d{2}-\d{2})T(\d{2})(\d{2})(\d{2})Z', filename)
    if aia_match:
        date_part = aia_match.group(1)
        h, m, s = aia_match.group(2), aia_match.group(3), aia_match.group(4)
        try:
            return datetime.strptime(f"{date_part}T{h}:{m}:{s}", '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            pass

    # HMI pattern: YYYYMMDD_HHMMSS_TAI
    hmi_match = re.search(r'(\d{8})_(\d{6})_TAI', filename)
    if hmi_match:
        date_part = hmi_match.group(1)
        time_part = hmi_match.group(2)
        try:
            return datetime.strptime(f"{date_part}_{time_part}", '%Y%m%d_%H%M%S')
        except ValueError:
            pass

    return None


def parse_fits_datetime(file_path: str) -> datetime | None:
    """Extract observation datetime from FITS header or filename.

    Supports various FITS header formats including:
    - DATE-OBS with ISO format (YYYY-MM-DDTHH:MM:SS)
    - DATE-OBS + TIME-OBS separate fields
    - T_OBS (SDO format)
    - Legacy DD/MM/YY format
    - Fallback to filename parsing for SDO files

    Args:
        file_path: Path to the FITS file.

    Returns:
        Datetime object or None if parsing failed.
    """
    if HAS_ASTROPY:
        try:
            with fits.open(file_path) as hdul:
                header = hdul[0].header

                # Try T_REC first (SDO standard), then T_OBS
                t_obs = header.get('T_REC') or header.get('T_OBS')
                if t_obs:
                    result = _parse_datetime_string(t_obs)
                    if result:
                        return result

                # Try DATE-OBS
                date_obs = header.get('DATE-OBS') or header.get('DATE_OBS')
                if date_obs:
                    time_obs = header.get('TIME-OBS') or header.get('TIME_OBS')
                    if time_obs:
                        combined = f"{date_obs} {time_obs}"
                        result = _parse_datetime_string(combined)
                    else:
                        result = _parse_datetime_string(date_obs)
                    if result:
                        return result

        except Exception:
            pass

    # Fallback: try to parse from filename
    return _parse_datetime_from_filename(file_path)
