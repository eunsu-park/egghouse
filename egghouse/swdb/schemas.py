"""Reference declarative schemas for solar-image instruments.

These are the canonical table specs (column → SQL type plus
``_primary_key`` / ``_unique`` / ``_indexes``) consumed by
``egghouse.database.create_tables_from_schema``. They are reference
*data*, not policy: the database name, credentials, and data roots
stay in the consuming project's config — only the table *shape* lives
here so consumers do not re-author DDL.

The shapes are byte-for-byte identical (verified) to the
``schema_config`` blocks in the setup-sw-db reference project, so a
project may build a setup-sw-db-compatible ``solar_images`` database
straight from these constants.
"""

from __future__ import annotations

from typing import Any, Dict

SDO_SCHEMA: Dict[str, Any] = {
    "telescope": "VARCHAR(10) NOT NULL",
    "channel": "VARCHAR(20) NOT NULL",
    "datetime": "TIMESTAMP NOT NULL",
    "file_path": "VARCHAR(512) NOT NULL",
    "quality": "INTEGER",
    "wavelength": "INTEGER",
    "exposure_time": "REAL",
    "_primary_key": ["telescope", "channel", "datetime"],
    "_unique": ["file_path"],
    "_indexes": [["datetime"], ["telescope"], ["quality"]],
}

LASCO_SCHEMA: Dict[str, Any] = {
    "camera": "VARCHAR(4) NOT NULL",
    "datetime": "TIMESTAMP NOT NULL",
    "file_path": "VARCHAR(512) NOT NULL",
    "exposure_time": "REAL",
    "filter": "VARCHAR(20)",
    "_primary_key": ["camera", "datetime"],
    "_unique": ["file_path"],
    "_indexes": [["datetime"]],
}

SECCHI_SCHEMA: Dict[str, Any] = {
    "datatype": "VARCHAR(10) NOT NULL",
    "spacecraft": "VARCHAR(10) NOT NULL",
    "instrument": "VARCHAR(10) NOT NULL",
    "channel": "VARCHAR(20)",
    "datetime": "TIMESTAMP NOT NULL",
    "file_path": "VARCHAR(512) NOT NULL",
    "exposure_time": "REAL",
    "filter": "VARCHAR(20)",
    "wavelength": "INTEGER",
    "_primary_key": ["datatype", "spacecraft", "instrument", "channel", "datetime"],
    "_unique": ["file_path"],
    "_indexes": [["datetime"], ["spacecraft"], ["instrument"]],
}
