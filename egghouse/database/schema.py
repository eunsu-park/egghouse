"""Declarative schema creation for PostgreSQL.

A *schema_config* is a plain dict mapping table name -> table spec. A
table spec maps column name -> SQL type, plus optional metadata keys:

    {
        "sdo": {
            "telescope":  "VARCHAR(10) NOT NULL",
            "channel":    "VARCHAR(20) NOT NULL",
            "datetime":   "TIMESTAMP NOT NULL",
            "file_path":  "VARCHAR(512) NOT NULL",
            "_primary_key": ["telescope", "channel", "datetime"],
            "_unique":      ["file_path"],
            "_indexes":     [["datetime"], ["telescope"]],
        },
        ...
    }

The format is **instrument-blind**: this module never inspects what a
table "means", only its declared columns and constraints. Feeding any
declarative config (solar images, space-weather time series, anything)
through :func:`initialize_database` creates exactly those tables.

The SQL builders are pure functions (no DB connection) so they are
unit-testable without a live PostgreSQL. The thin wrappers that open a
:class:`~egghouse.database.PostgresManager` are integration-level.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Union

# Reserved metadata keys inside a table spec.
_META_KEYS = ("_primary_key", "_unique", "_indexes")

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(name: str, *, kind: str) -> str:
    """Reject identifiers that are not plain `[A-Za-z_][A-Za-z0-9_]*`.

    Schema configs are developer-authored, not external input, but a
    typo that produced injectable DDL would be silent and dangerous, so
    this is a cheap boundary check rather than full SQL quoting.
    """
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(f"unsafe {kind} identifier: {name!r}")
    return name


def split_schema_meta(
    table_spec: Dict[str, Any],
) -> Tuple[Dict[str, str], Union[List[str], None], Union[List, None], Union[List, None]]:
    """Split a table spec into (columns, primary_key, unique, indexes).

    Non-mutating: the caller's dict is left untouched.
    """
    columns: Dict[str, str] = {}
    for key, value in table_spec.items():
        if key in _META_KEYS:
            continue
        columns[key] = value
    primary_key = table_spec.get("_primary_key")
    unique = table_spec.get("_unique")
    indexes = table_spec.get("_indexes")
    return columns, primary_key, unique, indexes


def build_create_table_sql(table_name: str, table_spec: Dict[str, Any]) -> str:
    """Pure: declarative table spec -> a single CREATE TABLE statement.

    - When ``_primary_key`` is given, any inline `PRIMARY KEY` is stripped
      from individual column types and a composite
      ``PRIMARY KEY (...)`` constraint is appended.
    - ``_unique`` entries may be a column name or a list of column names
      (multi-column UNIQUE).
    """
    _check_identifier(table_name, kind="table")
    columns, primary_key, unique, _ = split_schema_meta(table_spec)
    if not columns:
        raise ValueError(f"table {table_name!r} has no columns")

    parts: List[str] = []
    for col_name, col_type in columns.items():
        _check_identifier(col_name, kind="column")
        if primary_key:
            # Composite PK will be declared separately; drop inline PK.
            col_type = re.sub(r"\s+primary\s+key", "", col_type, flags=re.IGNORECASE)
        parts.append(f"{col_name} {col_type.strip()}")

    if primary_key:
        for c in primary_key:
            _check_identifier(c, kind="primary-key column")
        parts.append(f"PRIMARY KEY ({', '.join(primary_key)})")

    if unique:
        unique_specs = unique if isinstance(unique, (list, tuple)) else [unique]
        for spec in unique_specs:
            cols = [spec] if isinstance(spec, str) else list(spec)
            for c in cols:
                _check_identifier(c, kind="unique column")
            parts.append(f"UNIQUE ({', '.join(cols)})")

    return f"CREATE TABLE {table_name} ({', '.join(parts)})"


def build_index_sql(table_name: str, indexes: Union[List, None]) -> List[str]:
    """Pure: ``_indexes`` -> list of CREATE INDEX statements.

    Each entry is a column name or a list of column names.
    """
    _check_identifier(table_name, kind="table")
    if not indexes:
        return []
    statements: List[str] = []
    for entry in indexes:
        cols = [entry] if isinstance(entry, str) else list(entry)
        for c in cols:
            _check_identifier(c, kind="index column")
        idx_name = f"idx_{table_name}_{'_'.join(cols)}"
        statements.append(
            f"CREATE INDEX {idx_name} ON {table_name} ({', '.join(cols)})"
        )
    return statements


def _apply_schema(db, schema_config: Dict[str, Any], drop: bool, verbose: bool) -> Dict[str, str]:
    """Create tables on an already-open db-like object.

    `db` must provide `list_tables(names_only=True) -> list[str]` and
    `execute(sql)`. Returns {table_name: action} where action is one of
    "created", "recreated", "skipped".
    """
    existing = set(db.list_tables(names_only=True))
    actions: Dict[str, str] = {}
    for name, spec in schema_config.items():
        if name in existing:
            if drop:
                _check_identifier(name, kind="table")
                db.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
                actions[name] = "recreated"
            else:
                actions[name] = "skipped"
                if verbose:
                    print(f"  {name}: exists (skip)")
                continue
        else:
            actions[name] = "created"

        db.execute(build_create_table_sql(name, spec))
        _, _, _, indexes = split_schema_meta(spec)
        for stmt in build_index_sql(name, indexes):
            db.execute(stmt)
        if verbose:
            n_cols = len([k for k in spec if k not in _META_KEYS])
            print(f"  {name}: {actions[name]} ({n_cols} columns)")
    return actions


def create_tables_from_schema(
    db_config: Dict[str, Any],
    schema_config: Dict[str, Any],
    *,
    drop: bool = False,
    verbose: bool = False,
) -> Dict[str, str]:
    """Create all tables described by ``schema_config``.

    Args:
        db_config: kwargs for :class:`PostgresManager` (host, port,
            database, user, password, ...).
        schema_config: {table_name: table_spec}.
        drop: If True, existing tables are dropped (CASCADE) and rebuilt.
        verbose: Print per-table actions.

    Returns:
        {table_name: "created" | "recreated" | "skipped"}.
    """
    from .postgres import PostgresManager

    with PostgresManager(**db_config) as db:
        return _apply_schema(db, schema_config, drop=drop, verbose=verbose)


def create_database(db_config: Dict[str, Any], *, verbose: bool = False) -> bool:
    """Create the target database if it does not already exist.

    Tries to connect to the target database first; if that succeeds it
    already exists. Otherwise connects to an admin database
    (``template1`` then ``postgres``) and issues ``CREATE DATABASE``.

    Returns True if the database exists or was created, else False.
    """
    from .postgres import PostgresManager

    db_name = db_config["database"]
    _check_identifier(db_name, kind="database")

    try:
        with PostgresManager(**db_config):
            if verbose:
                print(f"✓ {db_name} already exists")
            return True
    except Exception:
        pass

    for admin_db in ("template1", "postgres"):
        try:
            admin_config = {**db_config, "database": admin_db}
            with PostgresManager(**admin_config) as db:
                exists = db.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    params=(db_name,),
                    fetch=True,
                )
                if not exists:
                    db.execute(f"CREATE DATABASE {db_name}")
                    if verbose:
                        print(f"✓ {db_name} created")
                elif verbose:
                    print(f"✓ {db_name} already exists")
                return True
        except Exception:
            continue

    if verbose:
        print(f"✗ {db_name} creation failed - create it manually")
    return False


def initialize_database(
    db_config: Dict[str, Any],
    schema_config: Dict[str, Any],
    *,
    verbose: bool = False,
) -> Dict[str, str]:
    """Create the database (idempotent) and all tables.

    Convenience wrapper combining :func:`create_database` and
    :func:`create_tables_from_schema`.
    """
    create_database(db_config, verbose=verbose)
    return create_tables_from_schema(db_config, schema_config, verbose=verbose)
