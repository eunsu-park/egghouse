"""Bulk record helpers: DataFrame upsert and orphan cleanup.

These sit on top of :class:`~egghouse.database.PostgresManager`. The
SQL-string builders and record normalization are pure functions
(unit-testable without a live PostgreSQL); the functions that open a
connection are integration-level.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(name: str, *, kind: str) -> str:
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(f"unsafe {kind} identifier: {name!r}")
    return name


def normalize_records(df) -> List[Dict[str, Any]]:
    """Pure: DataFrame -> list of dict rows with lowercased columns and
    NaN replaced by None (so psycopg2 writes SQL NULL)."""
    import pandas as pd

    if df.empty:
        return []
    df = df.copy()
    df.columns = df.columns.str.lower()
    records = df.to_dict("records")
    for rec in records:
        for k, v in rec.items():
            if pd.isna(v):
                rec[k] = None
    return records


def build_upsert_sql(
    table: str,
    columns: List[str],
    conflict_columns: Union[str, List[str]],
) -> str:
    """Pure: build an ``INSERT ... ON CONFLICT (...) DO NOTHING``.

    Supports composite conflict targets. A single positional-parameter
    row is expected per execute (``%s`` placeholders).
    """
    _check_identifier(table, kind="table")
    for c in columns:
        _check_identifier(c, kind="column")
    if isinstance(conflict_columns, str):
        conflict_columns = [conflict_columns]
    for c in conflict_columns:
        _check_identifier(c, kind="conflict column")

    cols = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    conflict = ", ".join(conflict_columns)
    return (
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO NOTHING"
    )


def upsert_dataframe(
    df,
    table: str,
    db_config: Dict[str, Any],
    *,
    conflict_columns: Union[str, List[str]] = "datetime",
    batch: int = 1000,
) -> int:
    """Insert rows, silently skipping conflicts on ``conflict_columns``.

    Idempotent: re-running with rows that already exist inserts nothing.
    Conflicts on *other* unique constraints (e.g. a ``UNIQUE(file_path)``
    separate from the composite PK) are also treated as skips rather
    than errors.

    Returns the number of rows actually inserted (excludes skips).
    """
    from .postgres import PostgresManager

    records = normalize_records(df)
    if not records:
        return 0
    columns = list(records[0].keys())
    sql = build_upsert_sql(table, columns, conflict_columns)

    inserted = 0
    with PostgresManager(**db_config) as db:
        for i in range(0, len(records), batch):
            for rec in records[i : i + batch]:
                values = tuple(rec[c] for c in columns)
                try:
                    db.execute(sql, params=values)
                    inserted += 1
                except Exception as exc:  # tolerate other UNIQUE violations
                    msg = str(exc).lower()
                    if "duplicate key" in msg or "unique" in msg:
                        continue
                    raise
    return inserted


def find_orphans(file_paths: List[str]) -> List[str]:
    """Pure-ish: return the subset of paths that no longer exist on disk."""
    return [p for p in file_paths if not Path(p).exists()]


def delete_orphans(
    table: str,
    db_config: Dict[str, Any],
    *,
    file_column: str = "file_path",
) -> int:
    """Delete rows whose referenced file no longer exists on disk.

    Returns the number of rows deleted.
    """
    from .postgres import PostgresManager

    _check_identifier(table, kind="table")
    _check_identifier(file_column, kind="column")

    with PostgresManager(**db_config) as db:
        rows = db.execute(f"SELECT {file_column} FROM {table}", fetch=True)
        if not rows:
            return 0
        all_paths = [r[file_column] for r in rows]
        orphans = find_orphans(all_paths)
        deleted = 0
        for path in orphans:
            db.execute(
                f"DELETE FROM {table} WHERE {file_column} = %s", params=(path,)
            )
            deleted += 1
        return deleted
