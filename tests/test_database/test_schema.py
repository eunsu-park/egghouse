"""Tests for egghouse.database.schema.

The pure SQL builders are tested directly. The connection-opening
wrappers (create_tables_from_schema / create_database /
initialize_database) require a live PostgreSQL and are integration-
level; instead, the table-creation *logic* is tested by driving
`_apply_schema` with a fake db that records executed SQL.
"""

from __future__ import annotations

import pytest

from egghouse.database.schema import (
    _apply_schema,
    build_create_table_sql,
    build_index_sql,
    split_schema_meta,
)


SDO_SPEC = {
    "telescope": "VARCHAR(10) NOT NULL",
    "channel": "VARCHAR(20) NOT NULL",
    "datetime": "TIMESTAMP NOT NULL",
    "file_path": "VARCHAR(512) NOT NULL",
    "quality": "INTEGER",
    "_primary_key": ["telescope", "channel", "datetime"],
    "_unique": ["file_path"],
    "_indexes": [["datetime"], ["telescope"]],
}


class TestSplitSchemaMeta:
    def test_separates_columns_from_metadata_without_mutating(self):
        spec = dict(SDO_SPEC)
        cols, pk, uniq, idx = split_schema_meta(spec)
        assert "telescope" in cols and "_primary_key" not in cols
        assert pk == ["telescope", "channel", "datetime"]
        assert uniq == ["file_path"]
        assert idx == [["datetime"], ["telescope"]]
        # Input dict untouched.
        assert spec == SDO_SPEC


class TestBuildCreateTableSql:
    def test_composite_primary_key_and_unique(self):
        sql = build_create_table_sql("sdo", SDO_SPEC)
        assert sql.startswith("CREATE TABLE sdo (")
        assert "telescope VARCHAR(10) NOT NULL" in sql
        assert "PRIMARY KEY (telescope, channel, datetime)" in sql
        assert "UNIQUE (file_path)" in sql

    def test_inline_primary_key_stripped_when_composite_present(self):
        spec = {
            "id": "SERIAL PRIMARY KEY",
            "datetime": "TIMESTAMP NOT NULL",
            "_primary_key": ["datetime"],
        }
        sql = build_create_table_sql("t", spec)
        # The inline 'PRIMARY KEY' on id must be removed; composite wins.
        assert "id SERIAL" in sql
        assert "SERIAL PRIMARY KEY" not in sql
        assert "PRIMARY KEY (datetime)" in sql

    def test_single_column_inline_pk_preserved_without_meta(self):
        spec = {"id": "SERIAL PRIMARY KEY", "name": "VARCHAR(50)"}
        sql = build_create_table_sql("t", spec)
        assert "id SERIAL PRIMARY KEY" in sql
        assert "PRIMARY KEY (" not in sql.replace("id SERIAL PRIMARY KEY", "")

    def test_multi_column_unique(self):
        spec = {
            "a": "INT",
            "b": "INT",
            "_primary_key": ["a"],
            "_unique": [["a", "b"]],
        }
        sql = build_create_table_sql("t", spec)
        assert "UNIQUE (a, b)" in sql

    def test_rejects_unsafe_table_identifier(self):
        with pytest.raises(ValueError, match="unsafe table"):
            build_create_table_sql("t; DROP TABLE x;--", {"a": "INT"})

    def test_rejects_unsafe_column_identifier(self):
        with pytest.raises(ValueError, match="unsafe column"):
            build_create_table_sql("t", {"a b": "INT"})

    def test_empty_columns_raises(self):
        with pytest.raises(ValueError, match="no columns"):
            build_create_table_sql("t", {"_primary_key": ["x"]})


class TestBuildIndexSql:
    def test_single_and_multi_column_indexes(self):
        stmts = build_index_sql("sdo", [["datetime"], ["telescope", "channel"]])
        assert "CREATE INDEX idx_sdo_datetime ON sdo (datetime)" in stmts
        assert (
            "CREATE INDEX idx_sdo_telescope_channel ON sdo (telescope, channel)"
            in stmts
        )

    def test_string_entry_supported(self):
        stmts = build_index_sql("t", ["datetime"])
        assert stmts == ["CREATE INDEX idx_t_datetime ON t (datetime)"]

    def test_none_or_empty_returns_empty(self):
        assert build_index_sql("t", None) == []
        assert build_index_sql("t", []) == []

    def test_rejects_unsafe_index_column(self):
        with pytest.raises(ValueError, match="unsafe index column"):
            build_index_sql("t", [["a); DROP"]])


class _FakeDB:
    """Records executed SQL; pretends a configurable set of tables exist."""

    def __init__(self, existing=()):
        self._existing = list(existing)
        self.executed: list[str] = []

    def list_tables(self, names_only=False):
        assert names_only is True
        return list(self._existing)

    def execute(self, sql, params=None, fetch=False):
        self.executed.append(sql)
        return None


class TestApplySchema:
    def test_creates_missing_tables_with_indexes(self):
        db = _FakeDB(existing=[])
        actions = _apply_schema(db, {"sdo": dict(SDO_SPEC)}, drop=False, verbose=False)
        assert actions == {"sdo": "created"}
        create = [s for s in db.executed if s.startswith("CREATE TABLE")]
        idx = [s for s in db.executed if s.startswith("CREATE INDEX")]
        assert len(create) == 1
        assert len(idx) == 2

    def test_skips_existing_table_when_not_drop(self):
        db = _FakeDB(existing=["sdo"])
        actions = _apply_schema(db, {"sdo": dict(SDO_SPEC)}, drop=False, verbose=False)
        assert actions == {"sdo": "skipped"}
        assert db.executed == []  # nothing run

    def test_drop_recreates_existing_table(self):
        db = _FakeDB(existing=["sdo"])
        actions = _apply_schema(db, {"sdo": dict(SDO_SPEC)}, drop=True, verbose=False)
        assert actions == {"sdo": "recreated"}
        assert any(s.startswith("DROP TABLE IF EXISTS sdo") for s in db.executed)
        assert any(s.startswith("CREATE TABLE sdo") for s in db.executed)

    def test_multiple_tables_mixed_state(self):
        db = _FakeDB(existing=["a"])
        cfg = {
            "a": {"x": "INT", "_primary_key": ["x"]},
            "b": {"y": "INT", "_primary_key": ["y"]},
        }
        actions = _apply_schema(db, cfg, drop=False, verbose=False)
        assert actions == {"a": "skipped", "b": "created"}
