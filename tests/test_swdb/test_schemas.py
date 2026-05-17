"""egghouse.swdb reference schemas must round-trip through the generic
declarative DDL builder and stay setup-sw-db compatible."""

from __future__ import annotations

from egghouse.database.schema import build_create_table_sql, build_index_sql
from egghouse.swdb import LASCO_SCHEMA, SDO_SCHEMA, SECCHI_SCHEMA


def test_sdo_schema_ddl():
    sql = build_create_table_sql("sdo", SDO_SCHEMA)
    assert sql == (
        "CREATE TABLE sdo (telescope VARCHAR(10) NOT NULL, "
        "channel VARCHAR(20) NOT NULL, datetime TIMESTAMP NOT NULL, "
        "file_path VARCHAR(512) NOT NULL, quality INTEGER, "
        "wavelength INTEGER, exposure_time REAL, "
        "PRIMARY KEY (telescope, channel, datetime), UNIQUE (file_path))"
    )
    idx = build_index_sql("sdo", SDO_SCHEMA["_indexes"])
    assert idx == [
        "CREATE INDEX idx_sdo_datetime ON sdo (datetime)",
        "CREATE INDEX idx_sdo_telescope ON sdo (telescope)",
        "CREATE INDEX idx_sdo_quality ON sdo (quality)",
    ]


def test_lasco_and_secchi_build_without_error():
    lasco = build_create_table_sql("lasco", LASCO_SCHEMA)
    assert "PRIMARY KEY (camera, datetime)" in lasco
    assert "UNIQUE (file_path)" in lasco
    secchi = build_create_table_sql("secchi", SECCHI_SCHEMA)
    assert (
        "PRIMARY KEY (datatype, spacecraft, instrument, channel, datetime)"
        in secchi
    )


def test_schemas_are_not_mutated_by_building():
    before = dict(SDO_SCHEMA)
    build_create_table_sql("sdo", SDO_SCHEMA)
    assert SDO_SCHEMA == before
