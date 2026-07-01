# egghouse.swdb User Guide

A DB domain layer for solar/space-weather image data. It converts FITS files into
validated metadata and DB records, and registers directories idempotently.

---

## Overview

`egghouse.swdb` is a **domain layer** that sits on top of the general-purpose
`egghouse.database` (v0.7) infrastructure (added in v0.8). Broken down by responsibility:

- **Reference schemas** (`SDO_SCHEMA`, `LASCO_SCHEMA`, `SECCHI_SCHEMA`) — declarative
  dicts that capture only the *shape* of the per-instrument tables. They are
  **byte-for-byte identical** (verified) to the `schema_config` block of the
  setup-sw-db reference project, so these constants alone are enough to build a
  setup-sw-db-compatible `solar_images` DB as-is. *Policy* such as the DB name,
  credentials, and data root stays in the consuming project's config.
- **`FitsHandler` ABC + `AiaFitsHandler`** — converts a single FITS file into a
  validated `ValidationResult`, a flat DB row, and an archive path.
- **`register_fits_dir`** — scans a directory tree → validates → idempotent upsert →
  (optionally) archives (moves) files.

Only **AIA is shipped** here (the scope undine needs). Other instruments such as
LASCO, SECCHI, and HMI are implemented in each project by subclassing `FitsHandler`.
`astropy` is **lazily imported** inside `extract_metadata`, so importing this
subpackage itself is lightweight and dependency-free.

```python
from egghouse.swdb import (
    SDO_SCHEMA, LASCO_SCHEMA, SECCHI_SCHEMA,
    ValidationResult,
    FitsHandler, AiaFitsHandler, AIA_EUV_WAVELENGTHS,
    scan_fits, register_fits_dir, RegisterReport,
)
```

---

## Reference Schemas

Each constant is a declarative dict made up of `{column: SQL type}` plus meta keys
(`_primary_key`, `_unique`, `_indexes`). It can be passed directly to
`egghouse.database.create_tables_from_schema` or
`egghouse.database.initialize_database`.

### SDO_SCHEMA

SDO/AIA image table. Uses `(telescope, channel, datetime)` as the primary key and
places a UNIQUE constraint on `file_path`. It is byte-for-byte identical to
setup-sw-db's `sdo` table.

```python
SDO_SCHEMA = {
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
```

### LASCO_SCHEMA

SOHO/LASCO coronagraph table. `(camera, datetime)` primary key.

```python
LASCO_SCHEMA = {
    "camera": "VARCHAR(4) NOT NULL",
    "datetime": "TIMESTAMP NOT NULL",
    "file_path": "VARCHAR(512) NOT NULL",
    "exposure_time": "REAL",
    "filter": "VARCHAR(20)",
    "_primary_key": ["camera", "datetime"],
    "_unique": ["file_path"],
    "_indexes": [["datetime"]],
}
```

### SECCHI_SCHEMA

STEREO/SECCHI table. `(datatype, spacecraft, instrument, channel, datetime)`
composite primary key.

```python
SECCHI_SCHEMA = {
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
```

### Creating a setup-sw-db-compatible sdo table

`egghouse.database.initialize_database` creates the DB idempotently and builds all
tables in the schema. Pass it in the form `{table_name: schema}`.

```python
from egghouse.database import initialize_database
from egghouse.swdb import SDO_SCHEMA

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "solar_images",
    "user": "username",
    "password": "password",
}

result = initialize_database(db_config, {"sdo": SDO_SCHEMA})
# result -> {"sdo": "created" | "recreated" | "skipped"}
```

The DDL of the generated `sdo` table is byte-for-byte identical to the
`schema_config` of the setup-sw-db reference, so it does not affect the setup-sw-db
side that git-pins egghouse.

---

## ValidationResult

A dataclass that holds the result of FITS validation + metadata extraction. Callers
check `result.success` instead of branching on something like
`isinstance(result, str)` (ported from setup-sw-db `core/result.py`).

### Fields

| Field | Type | Description |
|------|------|------|
| `success` | `bool` | Whether validation succeeded |
| `metadata` | `Optional[Dict[str, Any]]` | Extracted metadata on success, `None` on failure |
| `error` | `Optional[str]` | Error category on failure, `None` on success |
| `file_path` | `Optional[str]` | The file path the result refers to |

Error category (`error`) values: `invalid_file` (file could not be opened),
`invalid_header` (required header missing or non-AIA), `invalid_data` (no pixel
array / all NaN), `non_zero_quality` (`QUALITY != 0`).

### Constructor Methods

Rather than constructing `ValidationResult` directly, create it via two class methods.

```python
from egghouse.swdb import ValidationResult

ok = ValidationResult.ok({"datetime": ..., "telescope": "aia"}, "/data/a.fits")
# ok.success == True, ok.metadata == {...}, ok.error is None

bad = ValidationResult.fail("invalid_header", "/data/b.fits")
# bad.success == False, bad.error == "invalid_header", bad.metadata is None
```

#### Parameters

- `ValidationResult.ok(metadata, file_path=None)` — validation succeeded.
- `ValidationResult.fail(error, file_path=None)` — validation failed. `error` is one
  of the error category strings above.

---

## FitsHandler (ABC)

The per-instrument FITS processing interface. Three abstract methods must be
implemented.

| Method | Signature | Role |
|--------|----------|------|
| `extract_metadata` | `(file_path: str) -> ValidationResult` | Opens and validates the FITS file and returns the result |
| `to_db_record` | `(file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]` | Flattens validated metadata into a DB row dict |
| `target_dir` | `(root: str, metadata: Dict[str, Any]) -> Path` | Archive directory for the validated file (under `root`) |

By subclassing this ABC in each project for LASCO, SECCHI, HMI, etc., you can reuse
`register_fits_dir` as-is.

```python
from egghouse.swdb import FitsHandler

class MyInstrumentHandler(FitsHandler):
    def extract_metadata(self, file_path): ...
    def to_db_record(self, file_path, metadata): ...
    def target_dir(self, root, metadata): ...
```

---

## AiaFitsHandler

SDO/AIA EUV Level-1 handler. It is a concrete implementation of `FitsHandler`, and
only AIA is shipped.

```python
from egghouse.swdb import AiaFitsHandler

handler = AiaFitsHandler(check_data=False, require_quality_zero=False)
result = handler.extract_metadata("/data/aia.lev1_171a.fits")
if result.success:
    row = handler.to_db_record("/data/aia.lev1_171a.fits", result.metadata)
```

### Parameters

The constructor accepts keyword-only arguments:
`AiaFitsHandler(*, check_data=False, require_quality_zero=False)`.

- `check_data` (default `False`) — if `True`, fails with `invalid_data` when the
  pixel array is missing or all NaN. This forces reading the data, so it is slow. The
  default is header-only validation, which is fast for large batches.
- `require_quality_zero` (default `False`) — if `True`, fails with
  `non_zero_quality` when `QUALITY != 0`. The default registers everything and stores
  `quality` so you can filter later.

### Timestamp policy (intentional, documented divergence)

`AiaFitsHandler` uses the header's **`T_OBS` (UTC)** as the key, because it matches
undine's observation grouping. The setup-sw-db reference SDO validator uses `T_REC`
(JSOC slotted record time), but for AIA EUV `T_OBS` is the natural observation time
and requires no TAI conversion.

This is an **intentional, documented divergence**. The `sdo` *table shape* is still
compatible with setup-sw-db; only the semantic source of the `datetime` column
differs per project. `T_OBS` is parsed in `YYYY-MM-DDTHH:MM:SS.sssZ` format,
allowing a trailing `Z` and surrounding whitespace. The original string is preserved
in the metadata as `t_obs_raw`.

Compressed AIA lev1 files may have the image in HDU 1, so if HDU 1 has `T_OBS`, its
header/data is used; otherwise HDU 0 is used.

### The row produced by to_db_record

Returns a flat dict that matches `SDO_SCHEMA` exactly:

```python
{
    "telescope": "aia",
    "channel": "171",        # str(int(WAVELNTH))
    "datetime": <datetime>,  # T_OBS (UTC)
    "file_path": "/archive/aia/2026/20260514/aia.lev1_171a.fits",
    "quality": 0,
    "wavelength": 171,
    "exposure_time": 2.9,
}
```

`target_dir(root, metadata)` returns an archive path of the form
`<root>/aia/<YYYY>/<YYYYMMDD>/`.

---

## scan_fits

Recursively walks a directory tree and returns the list of FITS files (`List[Path]`).

```python
from egghouse.swdb import scan_fits

files = scan_fits("/data/incoming", pattern="*.fits", exclude_substrings=("spike",))
```

### Parameters

`scan_fits(scan_dir, *, pattern='*.fits', exclude_substrings=('spike',))`

- `scan_dir` — directory to scan recursively. Returns an empty list if it does not
  exist.
- `pattern` (default `'*.fits'`) — the glob pattern used by `rglob`.
- `exclude_substrings` (default `('spike',)`) — excludes files whose *name* contains
  any of these substrings (case-insensitive). The default filters out AIA `spike`
  artifact files.

Results are returned sorted.

---

## register_fits_dir

Scans a FITS tree → validates → idempotent upsert → (optionally) archives files, all
in one call, and returns a `RegisterReport`.

```python
from egghouse.swdb import register_fits_dir, AiaFitsHandler

report = register_fits_dir(
    "/data/incoming",
    handler=AiaFitsHandler(),
    table="sdo",
    db_config=db_config,
    conflict_columns=["telescope", "channel", "datetime"],
    move_root="/archive",
    error_dirs={"invalid_header": "_bad/header"},
    parallel=8,
    batch_size=1000,
    verbose=True,
)
print(report.summary())
```

### Parameters

`register_fits_dir(scan_dir, *, handler, table, db_config, conflict_columns,
move_root=None, error_dirs=None, pattern='*.fits',
exclude_substrings=('spike',), parallel=1, batch_size=1000, verbose=False)`

- `scan_dir` — directory to scan recursively.
- `handler` — a `FitsHandler` instance (e.g., `AiaFitsHandler()`).
- `table` — the target DB table name.
- `db_config` — a dict of kwargs passed to `PostgresManager`.
- `conflict_columns` — the composite conflict target for idempotent upsert (e.g.,
  `["telescope", "channel", "datetime"]`).
- `move_root` (default `None`) — if set, validated files are moved under
  `handler.target_dir(move_root, metadata)`. If the target already exists, it is
  counted as `skipped_existing` and the file is left in place. If `None`, files are
  registered where they are.
- `error_dirs` (default `None`) — `{error_category: subdirectory}`. Moves invalid
  files under `move_root`. Ignored if `move_root` is `None`.
- `pattern` / `exclude_substrings` — forwarded to `scan_fits`.
- `parallel` (default `1`) — number of thread workers for header validation. Since
  header-only validation is I/O-bound, a thread pool is used (no pickling
  constraints; the handler may carry state).
- `batch_size` (default `1000`) — the upsert batch size.
- `verbose` (default `False`) — prints per-file processing results.

### Idempotency

DB writes are delegated to `egghouse.database.upsert_dataframe`, which uses
`ON CONFLICT DO NOTHING`. Conflicts on both the composite PK of `SDO_SCHEMA` and the
separate `UNIQUE(file_path)` constraint are treated as skips rather than errors. So
re-running on an already-registered tree inserts nothing. File moves happen *only
after* the DB write for that batch has succeeded.

### Report Consistency

The counts in `RegisterReport` satisfy the following identity:

```
scanned == valid + sum(errors.values()) + skipped_existing
```

Every scanned file is attributed to exactly one place (valid / error category /
target-already-exists).

---

## RegisterReport

The dataclass returned by `register_fits_dir`.

| Field | Type | Description |
|------|------|------|
| `scanned` | `int` | Number of files scanned |
| `valid` | `int` | Number that passed validation and were turned into DB rows |
| `inserted` | `int` | Number of rows actually inserted into the DB (excluding skips) |
| `skipped_existing` | `int` | Number skipped because the move target already existed |
| `errors` | `Dict[str, int]` | Count per error category |

`inserted` is the *actual* number of inserts excluding idempotent skips, so on a
re-run `valid` may stay the same while `inserted` can be 0.

### .summary()

Returns a human-readable multiline string.

```python
print(report.summary())
# scanned files          : 1240
# valid                  : 1198
# inserted (DB)          : 1198
# skipped (target exists): 0
# errors:
#   invalid_header: 42
```

---

## End-to-End Example

The full flow of initializing the DB with `SDO_SCHEMA`, registering an AIA FITS
directory with `AiaFitsHandler`, and then printing the report.

```python
from egghouse.database import initialize_database
from egghouse.swdb import SDO_SCHEMA, AiaFitsHandler, register_fits_dir

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "solar_images",
    "user": "username",
    "password": "password",
}

# 1) Create the setup-sw-db-compatible sdo table (idempotent)
initialize_database(db_config, {"sdo": SDO_SCHEMA})

# 2) Scan the AIA FITS directory -> validate -> idempotent upsert -> archive move
report = register_fits_dir(
    "/data/incoming/aia",
    handler=AiaFitsHandler(check_data=False, require_quality_zero=False),
    table="sdo",
    db_config=db_config,
    conflict_columns=["telescope", "channel", "datetime"],
    move_root="/archive",
    error_dirs={
        "invalid_file": "_bad/file",
        "invalid_header": "_bad/header",
    },
    parallel=8,
    batch_size=1000,
)

# 3) Summarize the results
print(report.summary())
```

Running the same command again inserts no new rows (`inserted == 0`) thanks to the
idempotency of `register_fits_dir`, and the report counts remain consistent.
