"""
Database utilities for egghouse.

This package provides simple database management tools for research purposes.
"""

from .postgres import PostgresManager, to_dataframe
from .config import load_config, from_dict, create_example_config
from .schema import (
    build_create_table_sql,
    build_index_sql,
    split_schema_meta,
    create_tables_from_schema,
    create_database,
    initialize_database,
)
from .records import (
    normalize_records,
    build_upsert_sql,
    upsert_dataframe,
    find_orphans,
    delete_orphans,
)

__all__ = [
    'PostgresManager', 'to_dataframe',
    'load_config', 'from_dict', 'create_example_config',
    # Declarative schema
    'build_create_table_sql', 'build_index_sql', 'split_schema_meta',
    'create_tables_from_schema', 'create_database', 'initialize_database',
    # Bulk records
    'normalize_records', 'build_upsert_sql', 'upsert_dataframe',
    'find_orphans', 'delete_orphans',
]
__version__ = '0.2.0'
