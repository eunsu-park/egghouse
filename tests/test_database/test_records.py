"""Tests for egghouse.database.records.

Pure helpers (normalize_records, build_upsert_sql, find_orphans) are
tested directly. upsert_dataframe / delete_orphans open a live
PostgreSQL and are integration-level — their SQL/decision logic is
covered via the pure builders and a small filesystem check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from egghouse.database.records import (
    build_upsert_sql,
    find_orphans,
    normalize_records,
)


class TestNormalizeRecords:
    def test_lowercases_columns_and_nan_to_none(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(
            {"Datetime": ["2014-01-01"], "Quality": [np.nan], "Wave": [193]}
        )
        recs = normalize_records(df)
        assert recs == [{"datetime": "2014-01-01", "quality": None, "wave": 193}]

    def test_empty_dataframe_returns_empty_list(self):
        pd = pytest.importorskip("pandas")
        assert normalize_records(pd.DataFrame()) == []

    def test_does_not_mutate_input(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"A": [1]})
        normalize_records(df)
        assert list(df.columns) == ["A"]  # original casing preserved


class TestBuildUpsertSql:
    def test_single_conflict_column(self):
        sql = build_upsert_sql("sample", ["t_obs", "npz_path"], "t_obs")
        assert sql == (
            "INSERT INTO sample (t_obs, npz_path) VALUES (%s, %s) "
            "ON CONFLICT (t_obs) DO NOTHING"
        )

    def test_composite_conflict_columns(self):
        sql = build_upsert_sql(
            "sdo",
            ["telescope", "channel", "datetime", "file_path"],
            ["telescope", "channel", "datetime"],
        )
        assert "ON CONFLICT (telescope, channel, datetime) DO NOTHING" in sql
        assert sql.count("%s") == 4

    def test_rejects_unsafe_table(self):
        with pytest.raises(ValueError, match="unsafe table"):
            build_upsert_sql("x;DROP", ["a"], "a")

    def test_rejects_unsafe_column(self):
        with pytest.raises(ValueError, match="unsafe column"):
            build_upsert_sql("t", ["a b"], "a")

    def test_rejects_unsafe_conflict_column(self):
        with pytest.raises(ValueError, match="unsafe conflict column"):
            build_upsert_sql("t", ["a"], "a);--")


class TestFindOrphans:
    def test_returns_only_missing_paths(self, tmp_path: Path):
        present = tmp_path / "here.fits"
        present.write_bytes(b"")
        missing = str(tmp_path / "gone.fits")
        result = find_orphans([str(present), missing])
        assert result == [missing]

    def test_empty_input(self):
        assert find_orphans([]) == []
