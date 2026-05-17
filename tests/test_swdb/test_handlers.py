"""AiaFitsHandler tests against synthetic AIA-like FITS files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from egghouse.swdb import AiaFitsHandler, ValidationResult


def _write_aia_fits(
    path: Path,
    *,
    telescop="SDO/AIA",
    t_obs="2014-01-01T12:00:01.34Z",
    wavelnth=193,
    quality=0,
    exptime=2.9,
):
    """Minimal single-HDU AIA-like FITS. Pass None to omit a keyword."""
    from astropy.io import fits

    hdu = fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32))
    h = hdu.header
    if telescop is not None:
        h["TELESCOP"] = telescop
    if t_obs is not None:
        h["T_OBS"] = t_obs
    if wavelnth is not None:
        h["WAVELNTH"] = wavelnth
    if quality is not None:
        h["QUALITY"] = quality
    if exptime is not None:
        h["EXPTIME"] = exptime
    hdu.writeto(path, overwrite=True)
    return path


@pytest.fixture
def handler() -> AiaFitsHandler:
    return AiaFitsHandler()


class TestExtractMetadata:
    def test_valid_aia(self, handler, tmp_path):
        p = _write_aia_fits(tmp_path / "aia.fits")
        res = handler.extract_metadata(str(p))
        assert res.success
        m = res.metadata
        assert m["datetime"] == datetime(2014, 1, 1, 12, 0, 1, 340000)
        assert m["telescope"] == "aia"
        assert m["channel"] == "193"
        assert m["wavelength"] == 193
        assert m["quality"] == 0
        assert m["exposure_time"] == pytest.approx(2.9)

    def test_missing_t_obs_is_invalid_header(self, handler, tmp_path):
        p = _write_aia_fits(tmp_path / "x.fits", t_obs=None)
        res = handler.extract_metadata(str(p))
        assert not res.success and res.error == "invalid_header"

    def test_missing_wavelnth_is_invalid_header(self, handler, tmp_path):
        p = _write_aia_fits(tmp_path / "x.fits", wavelnth=None)
        res = handler.extract_metadata(str(p))
        assert not res.success and res.error == "invalid_header"

    def test_non_aia_telescope_is_invalid_header(self, handler, tmp_path):
        p = _write_aia_fits(tmp_path / "x.fits", telescop="SOHO")
        res = handler.extract_metadata(str(p))
        assert not res.success and res.error == "invalid_header"

    def test_unreadable_file_is_invalid_file(self, handler, tmp_path):
        bad = tmp_path / "not_fits.fits"
        bad.write_text("this is not a FITS file")
        res = handler.extract_metadata(str(bad))
        assert not res.success and res.error == "invalid_file"

    def test_quality_gate_optional(self, tmp_path):
        p = _write_aia_fits(tmp_path / "q.fits", quality=4)
        # default: register regardless of quality
        assert AiaFitsHandler().extract_metadata(str(p)).success
        # strict: non-zero quality rejected
        strict = AiaFitsHandler(require_quality_zero=True)
        res = strict.extract_metadata(str(p))
        assert not res.success and res.error == "non_zero_quality"

    def test_missing_exptime_yields_none(self, handler, tmp_path):
        p = _write_aia_fits(tmp_path / "ne.fits", exptime=None)
        res = handler.extract_metadata(str(p))
        assert res.success and res.metadata["exposure_time"] is None


class TestParseTObs:
    def test_trailing_z_and_fraction(self):
        assert AiaFitsHandler._parse_t_obs("2014-01-01T12:00:01.34Z") == datetime(
            2014, 1, 1, 12, 0, 1, 340000
        )

    def test_no_fraction(self):
        assert AiaFitsHandler._parse_t_obs("2020-06-15T00:00:00") == datetime(
            2020, 6, 15, 0, 0, 0
        )

    def test_garbage_returns_none(self):
        assert AiaFitsHandler._parse_t_obs("not-a-time") is None


class TestToDbRecordAndTargetDir:
    def test_to_db_record_shape(self, handler):
        meta = {
            "datetime": datetime(2014, 1, 1, 12, 0, 0),
            "telescope": "aia",
            "channel": "193",
            "wavelength": 193,
            "quality": 0,
            "exposure_time": 2.9,
        }
        rec = handler.to_db_record("/data/aia/x.fits", meta)
        assert rec == {
            "telescope": "aia",
            "channel": "193",
            "datetime": datetime(2014, 1, 1, 12, 0, 0),
            "file_path": "/data/aia/x.fits",
            "quality": 0,
            "wavelength": 193,
            "exposure_time": 2.9,
        }

    def test_target_dir_layout(self, handler):
        meta = {"datetime": datetime(2014, 3, 7, 5, 6, 7)}
        out = handler.target_dir("/archive", meta)
        assert out == Path("/archive/aia/2014/20140307")
