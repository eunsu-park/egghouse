"""register_fits_dir tests — the DB upsert is stubbed so no live
PostgreSQL is needed; the scan/validate/record/move orchestration is
what we verify."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from egghouse.swdb import AiaFitsHandler, register_fits_dir, scan_fits
from egghouse.swdb import register as register_mod


def _write_aia_fits(path: Path, *, t_obs, wavelnth, telescop="SDO/AIA"):
    from astropy.io import fits

    hdu = fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32))
    h = hdu.header
    h["TELESCOP"] = telescop
    h["T_OBS"] = t_obs
    h["WAVELNTH"] = wavelnth
    h["QUALITY"] = 0
    h["EXPTIME"] = 2.9
    hdu.writeto(path, overwrite=True)


@pytest.fixture
def captured_upserts(monkeypatch):
    """Replace egghouse.database.upsert_dataframe with a recorder.

    register_fits_dir does `from egghouse.database import
    upsert_dataframe` at call time, so patching the attribute on the
    egghouse.database module is sufficient.
    """
    calls = []

    def fake_upsert(df, table, db_config, *, conflict_columns, batch=1000):
        calls.append(
            {
                "table": table,
                "conflict_columns": list(conflict_columns),
                "records": df.to_dict("records"),
            }
        )
        return len(df)

    import egghouse.database as edb

    monkeypatch.setattr(edb, "upsert_dataframe", fake_upsert)
    return calls


class TestScanFits:
    def test_excludes_spike_and_recurses(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "aia_193.fits").write_bytes(b"")
        (tmp_path / "aia_171.spike.fits").write_bytes(b"")
        (tmp_path / "note.txt").write_bytes(b"")
        found = scan_fits(str(tmp_path))
        names = [p.name for p in found]
        assert "aia_193.fits" in names
        assert "aia_171.spike.fits" not in names
        assert "note.txt" not in names

    def test_missing_dir_returns_empty(self, tmp_path):
        assert scan_fits(str(tmp_path / "nope")) == []


class TestRegisterFitsDir:
    def test_registers_valid_and_counts_errors(self, tmp_path, captured_upserts):
        _write_aia_fits(tmp_path / "good1.fits", t_obs="2014-01-01T00:00:00Z", wavelnth=193)
        _write_aia_fits(tmp_path / "good2.fits", t_obs="2014-01-01T00:00:12Z", wavelnth=171)
        # invalid: not a FITS
        (tmp_path / "bad.fits").write_text("garbage")
        # invalid header: wrong telescope
        _write_aia_fits(tmp_path / "soho.fits", t_obs="2014-01-01T00:00:00Z",
                        wavelnth=195, telescop="SOHO")

        report = register_fits_dir(
            str(tmp_path),
            handler=AiaFitsHandler(),
            table="sdo",
            db_config={"database": "x"},
            conflict_columns=["telescope", "channel", "datetime"],
        )

        assert report.scanned == 4
        assert report.valid == 2
        assert report.inserted == 2
        assert report.errors.get("invalid_file") == 1
        assert report.errors.get("invalid_header") == 1
        # reconcile
        assert report.scanned == (
            report.valid + sum(report.errors.values()) + report.skipped_existing
        )

        assert len(captured_upserts) == 1
        call = captured_upserts[0]
        assert call["table"] == "sdo"
        assert call["conflict_columns"] == ["telescope", "channel", "datetime"]
        chans = sorted(r["channel"] for r in call["records"])
        assert chans == ["171", "193"]
        # no move_root -> file_path stays at the scanned location
        for r in call["records"]:
            assert str(tmp_path) in r["file_path"]

    def test_move_root_relocates_valid_files(self, tmp_path, captured_upserts):
        src = tmp_path / "incoming"
        src.mkdir()
        _write_aia_fits(src / "a.fits", t_obs="2014-03-07T01:02:03Z", wavelnth=211)
        archive = tmp_path / "archive"

        report = register_fits_dir(
            str(src),
            handler=AiaFitsHandler(),
            table="sdo",
            db_config={"database": "x"},
            conflict_columns=["telescope", "channel", "datetime"],
            move_root=str(archive),
        )
        assert report.valid == 1
        moved = archive / "aia" / "2014" / "20140307" / "a.fits"
        assert moved.is_file()
        assert not (src / "a.fits").exists()
        # DB record points at the post-move path
        assert captured_upserts[0]["records"][0]["file_path"] == str(moved)

    def test_empty_dir_no_db_call(self, tmp_path, captured_upserts):
        report = register_fits_dir(
            str(tmp_path),
            handler=AiaFitsHandler(),
            table="sdo",
            db_config={"database": "x"},
            conflict_columns=["telescope", "channel", "datetime"],
        )
        assert report.scanned == 0
        assert captured_upserts == []

    def test_parallel_path_equivalent(self, tmp_path, captured_upserts):
        for i in range(5):
            _write_aia_fits(
                tmp_path / f"f{i}.fits",
                t_obs=f"2014-01-01T00:00:{i:02d}Z",
                wavelnth=193 + i,
            )
        report = register_fits_dir(
            str(tmp_path),
            handler=AiaFitsHandler(),
            table="sdo",
            db_config={"database": "x"},
            conflict_columns=["telescope", "channel", "datetime"],
            parallel=4,
        )
        assert report.scanned == 5 and report.valid == 5
