"""Instrument-specific FITS handlers.

A handler turns a FITS file on disk into (1) validated metadata and
(2) a flat DB record matching the instrument's table schema, and
decides the on-disk archive location.

`FitsHandler` is the ABC; `AiaFitsHandler` is the concrete SDO/AIA
implementation. Other instruments (LASCO, SECCHI, HMI, …) subclass
`FitsHandler` in their own projects — only AIA ships here, which is
what undine needs.

`astropy` is imported lazily inside `extract_metadata`, so importing
this module is cheap and dependency-free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .result import ValidationResult

# SDO/AIA EUV passbands undine cares about.
AIA_EUV_WAVELENGTHS = (94, 131, 171, 193, 211, 304, 335)


class FitsHandler(ABC):
    """Common interface for instrument-specific FITS handling."""

    @abstractmethod
    def extract_metadata(self, file_path: str) -> ValidationResult:
        """Open the FITS, validate, and return a `ValidationResult`."""
        ...

    @abstractmethod
    def to_db_record(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten validated metadata into a DB-ready row dict."""
        ...

    @abstractmethod
    def target_dir(self, root: str, metadata: Dict[str, Any]) -> Path:
        """Archive directory for a validated file (under ``root``)."""
        ...


class AiaFitsHandler(FitsHandler):
    """SDO/AIA EUV Level-1 handler.

    Timestamp policy: uses ``T_OBS`` (UTC), consistent with undine's
    acquisition grouping. (The setup-sw-db reference uses ``T_REC``,
    the JSOC slotted record time; for AIA EUV ``T_OBS`` is the natural
    observation timestamp and needs no TAI conversion. This is a
    deliberate, documented divergence — the `sdo` *table shape* is
    still setup-sw-db compatible; only the column's semantic source
    differs per project.)

    Args:
        check_data: If True, also fail with ``invalid_data`` when the
            pixel array is missing / all-NaN (slower: forces a read).
            Default False — header-only validation, fast for large
            batches.
        require_quality_zero: If True, fail with ``non_zero_quality``
            when ``QUALITY != 0``. Default False (register everything;
            quality is stored for later filtering).
    """

    def __init__(self, *, check_data: bool = False, require_quality_zero: bool = False) -> None:
        self.check_data = check_data
        self.require_quality_zero = require_quality_zero

    def extract_metadata(self, file_path: str) -> ValidationResult:
        from astropy.io import fits

        try:
            hdul = fits.open(file_path)
        except Exception:
            return ValidationResult.fail("invalid_file", file_path)

        try:
            # AIA lev1: compressed image in HDU 1; header may live there.
            if len(hdul) > 1 and hdul[1].header.get("T_OBS"):
                header = hdul[1].header
                data_hdu = hdul[1]
            else:
                header = hdul[0].header
                data_hdu = hdul[0]

            t_obs = header.get("T_OBS") or header.get("t_obs")
            wavelnth = header.get("WAVELNTH")
            if wavelnth is None:
                wavelnth = header.get("wavelnth")
            telescop = header.get("TELESCOP") or header.get("telescop")

            if t_obs is None or wavelnth is None or telescop is None:
                return ValidationResult.fail("invalid_header", file_path)
            if "AIA" not in str(telescop).upper():
                return ValidationResult.fail("invalid_header", file_path)

            dt = self._parse_t_obs(str(t_obs))
            if dt is None:
                return ValidationResult.fail("invalid_header", file_path)

            if self.check_data:
                import numpy as np

                try:
                    data = data_hdu.data
                    if data is None or data.size == 0 or np.all(np.isnan(data)):
                        return ValidationResult.fail("invalid_data", file_path)
                except Exception:
                    return ValidationResult.fail("invalid_data", file_path)

            quality = header.get("QUALITY")
            if quality is None:
                quality = header.get("quality")
            if self.require_quality_zero and quality not in (0, None) and quality != 0:
                return ValidationResult.fail("non_zero_quality", file_path)

            exptime = header.get("EXPTIME")
            if exptime is None:
                exptime = header.get("exptime")

            return ValidationResult.ok(
                {
                    "datetime": dt,
                    "telescope": "aia",
                    "channel": str(int(wavelnth)),
                    "wavelength": int(wavelnth),
                    "quality": int(quality) if quality is not None else None,
                    "exposure_time": float(exptime) if exptime is not None else None,
                    "t_obs_raw": str(t_obs),
                },
                file_path,
            )
        except Exception:
            return ValidationResult.fail("invalid_header", file_path)
        finally:
            hdul.close()

    @staticmethod
    def _parse_t_obs(t_obs: str) -> "datetime | None":
        """Parse the AIA ``T_OBS`` string (UTC).

        AIA writes ``YYYY-MM-DDTHH:MM:SS.sssZ``; tolerate a trailing
        ``Z`` and surrounding whitespace.
        """
        s = t_obs.strip().rstrip("Z").rstrip()
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None

    def to_db_record(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "telescope": metadata["telescope"],
            "channel": metadata["channel"],
            "datetime": metadata["datetime"],
            "file_path": str(file_path),
            "quality": metadata.get("quality"),
            "wavelength": metadata.get("wavelength"),
            "exposure_time": metadata.get("exposure_time"),
        }

    def target_dir(self, root: str, metadata: Dict[str, Any]) -> Path:
        dt: datetime = metadata["datetime"]
        return Path(root) / "aia" / f"{dt.year:04d}" / f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"
