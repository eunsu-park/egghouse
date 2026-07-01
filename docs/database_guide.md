# egghouse.database Usage Guide

PostgreSQL database management utility.

> **v0.7+ / v0.8+**: For declarative schema creation and bulk record helpers, see the
> [Declarative Schema](#declarative-schema-v07) and
> [Bulk Record Helpers](#bulk-record-helpers-v07) sections below.
> For the solar / space-weather DB domain layer (`egghouse.swdb`, v0.8), see the
> [egghouse.swdb Integration](#egghouseswdb-integration-v08) section.

---

## Overview

A simple PostgreSQL management tool for research purposes:
- Database/schema/table management
- CRUD operations (Insert, Select, Update, Delete)
- Upsert (Insert on Conflict)
- Date range queries
- pandas DataFrame conversion

---

## Installation

```bash
pip install psycopg2-binary
```

---

## Configuration

### Environment Variables

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=solar_data       # or DB_DATABASE
export DB_USER=username
export DB_PASSWORD=password
export DB_LOG_QUERIES=true      # query logging (optional)
```

```python
from egghouse.database import PostgresManager, load_config

config = load_config()  # loaded automatically from environment variables
db = PostgresManager(**config['database'])
```

### YAML File

```yaml
# database.yaml
database:
  host: localhost
  port: 5432
  database: solar_data
  user: username
  password: password
  log_queries: true
```

```python
config = load_config('database.yaml')
db = PostgresManager(**config['database'])
```

### JSON File

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "solar_data",
    "user": "username",
    "password": "password"
  }
}
```

```python
config = load_config('database.json')
db = PostgresManager(**config['database'])
```

### Direct Specification

```python
from egghouse.database import PostgresManager, from_dict

# create from a dictionary
config = from_dict({
    'host': 'localhost',
    'port': 5432,
    'database': 'solar_data',
    'user': 'username',
    'password': 'password'
})
db = PostgresManager(**config)

# or pass arguments directly
db = PostgresManager(
    host='localhost',
    port=5432,
    database='solar_data',
    user='username',
    password='password',
    log_queries=True
)
```

### Creating an Example Config File

```python
from egghouse.database import create_example_config

create_example_config('config.example.yaml')
```

---

## PostgresManager Usage

### Connection Management

```python
from egghouse.database import PostgresManager

# basic connection
db = PostgresManager(
    host='localhost',
    database='solar_data',
    user='user',
    password='pass'
)

# close after finishing work
db.close()

# using a context manager (recommended)
with PostgresManager(**config) as db:
    db.insert('users', {'name': 'test'})
    # close() is called automatically
```

---

## Data Manipulation (CRUD)

### Insert

```python
# insert a single row
db.insert('users', {'name': 'Eunsu', 'email': 'eunsu@kasi.re.kr'})

# insert multiple rows
db.insert('users', [
    {'name': 'User1', 'email': 'user1@example.com'},
    {'name': 'User2', 'email': 'user2@example.com'}
])

# return the ID (when there is a SERIAL column)
new_id = db.insert('users', {'name': 'New'}, return_id=True)
```

### Select

```python
# select all rows
users = db.select('users')

# select specific columns only
users = db.select('users', columns=['id', 'name'])

# WHERE condition
users = db.select('users', where={'name': 'Eunsu'})

# ordering and limit
users = db.select('users', order_by='created_at DESC', limit=10)

# combined
users = db.select('users',
    columns=['id', 'name', 'email'],
    where={'active': True},
    order_by='name ASC',
    limit=100
)
```

### Update

```python
# WHERE clause is required!
affected_rows = db.update('users',
    data={'email': 'new@example.com'},
    where={'name': 'Eunsu'}
)
print(f"Updated {affected_rows} rows")
```

### Delete

```python
# WHERE clause is required!
deleted_rows = db.delete('users', where={'name': 'Eunsu'})
print(f"Deleted {deleted_rows} rows")
```

---

## Upsert (Insert or Update)

INSERT ON CONFLICT DO UPDATE, which updates on conflict.

```python
# single-row upsert
db.upsert('observations',
    data={'filepath': '/data/aia.fits', 'wavelength': 171, 'processed': True},
    conflict_columns='filepath',
    update_columns=['processed']
)

# multi-row upsert
db.upsert('observations', [
    {'filepath': f1, 'wavelength': 171, 'status': 'done'},
    {'filepath': f2, 'wavelength': 193, 'status': 'done'}
], conflict_columns='filepath')

# composite key conflict
db.upsert('data',
    data={'date': '2024-01-01', 'wavelength': 171, 'count': 10},
    conflict_columns=['date', 'wavelength']
)

# when update_columns is not specified, all columns except conflict_columns are updated
db.upsert('users',
    data={'username': 'eunsu', 'email': 'new@email.com', 'status': 'active'},
    conflict_columns='username'
)  # email and status get updated
```

---

## Date Range Queries

```python
from datetime import datetime

start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)

# default (start <= date < end)
results = db.select_date_range('observations',
    date_column='date',
    start_date=start,
    end_date=end
)

# include end date (start <= date <= end)
results = db.select_date_range('observations',
    date_column='date',
    start_date=start,
    end_date=end,
    inclusive_end=True
)

# with additional conditions
results = db.select_date_range('observations',
    date_column='timestamp',
    start_date=start,
    end_date=end,
    columns=['id', 'filepath', 'wavelength'],
    where={'wavelength': 171},
    order_by='timestamp DESC',
    limit=1000
)
```

---

## Table Management

### Creating a Table

```python
db.create_table('observations', {
    'id': 'SERIAL PRIMARY KEY',
    'filepath': 'VARCHAR(500) UNIQUE NOT NULL',
    'wavelength': 'INTEGER',
    'date': 'TIMESTAMP',
    'processed': 'BOOLEAN DEFAULT FALSE',
    'created_at': 'TIMESTAMP DEFAULT NOW()'
})
```

### Listing Tables

```python
# full information (name, size)
tables = db.list_tables()
# [{'name': 'users', 'size': '8192 bytes'}, ...]

# names only
table_names = db.list_tables(names_only=True)
# ['users', 'observations', ...]
```

### Inspecting Table Structure

```python
columns = db.describe_table('observations')
for col in columns:
    print(f"{col['name']}: {col['type']}")
```

### Checking Table Existence

```python
if db.table_exists('observations'):
    print("Table exists")
```

### Dropping a Table

```python
db.drop_table('old_table')
db.drop_table('parent_table', cascade=True)  # also drop dependent objects
```

---

## Utilities

### Counting Rows

```python
total = db.count('observations')
filtered = db.count('observations', where={'wavelength': 171})
```

### Emptying a Table

```python
db.truncate('temp_data')
db.truncate('parent_table', cascade=True)
```

### VACUUM

```python
db.vacuum()                        # entire database
db.vacuum('observations')          # specific table
db.vacuum('observations', full=True)  # VACUUM FULL
```

---

## DataFrame Conversion

```python
from egghouse.database import to_dataframe

results = db.select('observations')
df = to_dataframe(results)

# parse date columns
df = to_dataframe(results, parse_dates=['date', 'created_at'])

# direct use
import pandas as pd
df = pd.DataFrame(results)
```

---

## Raw SQL Execution

```python
# query with no result
db.execute("CREATE INDEX idx_wavelength ON observations(wavelength)")

# query with a result
results = db.execute(
    "SELECT * FROM observations WHERE wavelength = %s",
    params=(171,),
    fetch=True
)

# parameterized query (prevents SQL injection)
results = db.execute(
    "SELECT * FROM users WHERE name = %s AND status = %s",
    params=('Eunsu', 'active'),
    fetch=True
)
```

---

## Schema Management

```python
# create a schema
db.create_schema('solar')

# create a table within a schema
db.create_table('observations', {...}, schema='solar')

# manipulate tables within a schema
db.insert('observations', data, schema='solar')
db.select('observations', schema='solar')

# list schemas
schemas = db.list_schemas()

# drop a schema
db.drop_schema('solar', cascade=True)
```

---

## Database Management

```python
# connect without a database (for administration)
db = PostgresManager(host='localhost', user='admin', password='pass')

# create a database
db.create_database('new_db')

# list databases
databases = db.list_databases()

# drop a database
db.drop_database('old_db', force=True)  # forcibly terminate connections then drop
```

---

## SQL Injection Prevention

PostgresManager uses the `psycopg2.sql` module to prevent SQL Injection:

```python
# safe: use a parameterized query
db.select('users', where={'name': user_input})

# safe: handle table/column names with sql.Identifier
db.insert(table_name, data)  # uses sql.Identifier internally

# dangerous: direct string concatenation (do not do this!)
# db.execute(f"SELECT * FROM {table_name}")  # NEVER DO THIS
```

---

## Full Example

```python
from datetime import datetime, timedelta
from egghouse.database import PostgresManager, load_config, to_dataframe

# load config
config = load_config('database.yaml')

with PostgresManager(**config['database']) as db:
    # create table
    if not db.table_exists('observations'):
        db.create_table('observations', {
            'id': 'SERIAL PRIMARY KEY',
            'filepath': 'VARCHAR(500) UNIQUE NOT NULL',
            'wavelength': 'INTEGER',
            'date': 'TIMESTAMP',
            'processed': 'BOOLEAN DEFAULT FALSE'
        })

    # insert data
    db.insert('observations', {
        'filepath': '/data/aia_171_20240101.fits',
        'wavelength': 171,
        'date': datetime(2024, 1, 1, 12, 0, 0)
    })

    # Upsert (update on duplicate)
    db.upsert('observations', {
        'filepath': '/data/aia_171_20240101.fits',
        'wavelength': 171,
        'date': datetime(2024, 1, 1, 12, 0, 0),
        'processed': True
    }, conflict_columns='filepath')

    # query
    today = datetime.now()
    week_ago = today - timedelta(days=7)

    results = db.select_date_range('observations',
        date_column='date',
        start_date=week_ago,
        end_date=today,
        where={'wavelength': 171}
    )

    # DataFrame conversion
    df = to_dataframe(results, parse_dates=['date'])
    print(f"Found {len(df)} observations")
```

---

## Declarative Schema (v0.7+)

Whereas `PostgresManager.create_table()` creates tables one at a time,
the `egghouse.database.schema` module **declaratively creates an entire schema
from a single config dictionary**. It is infrastructure generalized and imported
from setup-sw-db's `core/database.py`, letting you build a schema with just a
config dict + import, without any project-specific domain code.

### schema_config Format

`schema_config` is a plain dict of `table name -> table spec`.
A table spec is a `column name -> SQL type` mapping, plus the following optional
reserved meta keys:

| Meta key | Meaning |
|---------|------|
| `_primary_key` | List of columns that make up a composite PRIMARY KEY |
| `_unique` | UNIQUE constraint. A column name (str) or a list of columns (multi-column). If there are several, a list of lists |
| `_indexes` | Targets for CREATE INDEX. Each item is a column name or a list of columns |

```python
schema = {
    "sdo": {
        "telescope":    "VARCHAR(10) NOT NULL",
        "channel":      "VARCHAR(20) NOT NULL",
        "datetime":     "TIMESTAMP NOT NULL",
        "file_path":    "VARCHAR(512) NOT NULL",
        "_primary_key": ["telescope", "channel", "datetime"],
        "_unique":      ["file_path"],
        "_indexes":     [["datetime"], ["telescope"]],
    },
}
```

This format is **instrument-blind**. The module never looks into "what a table
means"; it only handles the declared columns and constraints. Whether solar
images or space-weather time series, if you pass an arbitrary declarative config
to `initialize_database`, exactly those tables are created.

> **Identifier safety**: Table/column/index identifiers are validated against the
> `^[A-Za-z_][A-Za-z0-9_]*$` pattern, and a violation raises `ValueError`.
> The schema config is written by the developer, but if a typo produces injectable
> DDL it becomes silently dangerous, so a boundary check is in place. Column *types*
> are free-form text.

### Pure Builders (no DB connection required)

The following functions are pure functions and can be unit-tested without a live
PostgreSQL.

#### build_create_table_sql

```python
from egghouse.database import build_create_table_sql

sql = build_create_table_sql("sdo", schema["sdo"])
# CREATE TABLE sdo (telescope VARCHAR(10) NOT NULL, ...,
#   PRIMARY KEY (telescope, channel, datetime), UNIQUE (file_path))
```

Converts a declarative table spec into a single `CREATE TABLE` statement.
When `_primary_key` is given, the inline `PRIMARY KEY` on individual column types
is removed and a composite `PRIMARY KEY (...)` constraint is appended at the end.
An `_unique` item can be a column name or a list of column names (multi-column
UNIQUE).

##### Parameters
- `table_name` (str): Table name (identifier is validated).
- `table_spec` (dict): Column mapping + meta keys. If there are no columns, `ValueError`.

#### build_index_sql

```python
from egghouse.database import build_index_sql

stmts = build_index_sql("sdo", [["datetime"], ["telescope"]])
# ['CREATE INDEX idx_sdo_datetime ON sdo (datetime)',
#  'CREATE INDEX idx_sdo_telescope ON sdo (telescope)']
```

Converts `_indexes` into a list of `CREATE INDEX` statements. Each item is a
column name or a list of column names. Index names are generated following the
`idx_<table>_<cols>` convention.

##### Parameters
- `table_name` (str): Table name.
- `indexes` (list | None): The `_indexes` value. Returns an empty list if empty.

#### split_schema_meta

```python
from egghouse.database import split_schema_meta

columns, primary_key, unique, indexes = split_schema_meta(schema["sdo"])
# columns = {'telescope': 'VARCHAR(10) NOT NULL', ...}
# primary_key = ['telescope', 'channel', 'datetime']
```

Splits a table spec into a `(columns, primary_key, unique, indexes)` tuple.
It does not modify the caller's dict (non-mutating). If a meta key is absent, the
corresponding value is `None`.

##### Parameters
- `table_spec` (dict): Table spec.

### Connection Wrappers (use a DB connection)

The following functions are integration-level functions that internally open a
`PostgresManager` and apply changes to the actual DB.

#### create_database

```python
from egghouse.database import create_database

ok = create_database(db_config, verbose=True)
```

Creates the target database if it does not exist (idempotent). It first tries to
connect to the target DB; if it succeeds, the DB already exists. If it fails, it
connects to an administrative DB (`template1` → `postgres`) and runs
`CREATE DATABASE`. Returns `True` if the DB exists or is created, `False` on
failure.

##### Parameters
- `db_config` (dict): `PostgresManager` arguments (host, port, database, user, password, ...).
- `verbose` (bool, keyword-only): Print progress. Default `False`.

#### create_tables_from_schema

```python
from egghouse.database import create_tables_from_schema

actions = create_tables_from_schema(db_config, schema, drop=False, verbose=True)
# {'sdo': 'created'}  # or 'recreated' / 'skipped'
```

Creates all tables described in `schema_config`. Existing tables are skipped
unless `drop=True` (`skipped`). If `drop=True`, existing tables are dropped with
`CASCADE` and recreated (`recreated`).

##### Parameters
- `db_config` (dict): `PostgresManager` arguments.
- `schema_config` (dict): `{table name: table spec}`.
- `drop` (bool, keyword-only): Drop existing tables and rebuild. Default `False`.
- `verbose` (bool, keyword-only): Print per-table action. Default `False`.

Return value: `{table name: "created" | "recreated" | "skipped"}`.

#### initialize_database

```python
from egghouse.database import initialize_database

schema = {
    "sdo": {
        "telescope":    "VARCHAR(10) NOT NULL",
        "channel":      "VARCHAR(20) NOT NULL",
        "datetime":     "TIMESTAMP NOT NULL",
        "file_path":    "VARCHAR(512) NOT NULL",
        "_primary_key": ["telescope", "channel", "datetime"],
        "_unique":      ["file_path"],
        "_indexes":     [["datetime"], ["telescope"]],
    },
}

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "solar_data",
    "user": "username",
    "password": "password",
}

actions = initialize_database(db_config, schema, verbose=True)
```

A convenience wrapper that combines `create_database` and
`create_tables_from_schema`. It (idempotently) creates the DB, then creates all
tables and returns the action dict.

##### Parameters
- `db_config` (dict): `PostgresManager` arguments.
- `schema_config` (dict): `{table name: table spec}`.
- `verbose` (bool, keyword-only): Print progress. Default `False`.

---

## Bulk Record Helpers (v0.7+)

The `egghouse.database.records` module provides bulk upsert of pandas DataFrames
and cleanup of orphan records. The SQL string builders and record normalization
are pure functions (unit-testable without PostgreSQL), while functions that open
a connection are integration-level.

#### normalize_records

```python
from egghouse.database import normalize_records

records = normalize_records(df)
# [{'telescope': 'sdo', 'datetime': ..., ...}, ...]
```

Converts a DataFrame into a list of dict rows. Column names are lowercased, and
`NaN` is replaced with `None` (so psycopg2 writes SQL `NULL`). An empty DataFrame
returns an empty list (handling an edge case where the original setup-sw-db code
crashed).

##### Parameters
- `df`: pandas DataFrame.

#### build_upsert_sql

```python
from egghouse.database import build_upsert_sql

sql = build_upsert_sql("sdo", ["telescope", "channel", "datetime", "file_path"],
                       ["telescope", "channel", "datetime"])
# INSERT INTO sdo (telescope, channel, datetime, file_path)
#   VALUES (%s, %s, %s, %s)
#   ON CONFLICT (telescope, channel, datetime) DO NOTHING
```

A pure function that generates an `INSERT ... ON CONFLICT (...) DO NOTHING`
statement. It supports composite conflict targets and expects a single positional
parameter row (`%s` placeholders) per execute. Identifiers are validated.

##### Parameters
- `table` (str): Table name.
- `columns` (list[str]): List of columns to insert.
- `conflict_columns` (str | list[str]): Columns used to detect conflicts. A single column if a string.

#### upsert_dataframe

```python
from egghouse.database import upsert_dataframe

inserted = upsert_dataframe(
    df, "sdo", db_config,
    conflict_columns=["telescope", "channel", "datetime"],
    batch=1000,
)
print(f"{inserted} rows inserted")

# idempotency: running again with the same data inserts nothing
again = upsert_dataframe(df, "sdo", db_config,
                         conflict_columns=["telescope", "channel", "datetime"])
assert again == 0
```

Inserts rows but silently skips `conflict_columns` conflicts. Because it is
**idempotent**, re-running with already-existing rows inserts nothing. Violations
of other UNIQUE constraints distinct from the composite PK (e.g.,
`UNIQUE(file_path)`) are also treated as skips rather than errors. Returns the
number of rows actually inserted (excluding skips).

##### Parameters
- `df`: pandas DataFrame.
- `table` (str): Target table.
- `db_config` (dict): `PostgresManager` arguments.
- `conflict_columns` (str | list[str], keyword-only): Columns used to detect conflicts. Default `"datetime"`.
- `batch` (int, keyword-only): Batch size. Default `1000`.

#### find_orphans

```python
from egghouse.database import find_orphans

missing = find_orphans(["/data/a.fits", "/data/b.fits"])
# returns only paths that no longer exist on disk
```

Returns the subset of the given paths that no longer exist on disk.

##### Parameters
- `file_paths` (list[str]): List of paths to check.

#### delete_orphans

```python
from egghouse.database import delete_orphans

deleted = delete_orphans("sdo", db_config, file_column="file_path")
print(f"{deleted} orphan rows deleted")
```

Deletes rows whose referenced file has disappeared from disk. Returns the number
of deleted rows.

##### Parameters
- `table` (str): Target table.
- `db_config` (dict): `PostgresManager` arguments.
- `file_column` (str, keyword-only): File path column name. Default `"file_path"`.

---

## egghouse.swdb Integration (v0.8+)

v0.8's `egghouse.swdb` is a solar / space-weather DB domain layer built on top of
the generic (instrument-blind) helpers above. It reuses this module's declarative
schema and bulk upsert as-is, while adding domain features such as reference
schemas (`SDO_SCHEMA`, `LASCO_SCHEMA`, `SECCHI_SCHEMA`), FITS handlers
(`FitsHandler` ABC, AIA implementation), and directory scanning/registration.
For the internal behavior and usage examples of swdb, see `docs/swdb_guide.md`
and the Modules section of the root `README.MD`. Function signatures are in
`API_REFERENCE.md`, and the change history is in `CHANGELOG.md`.

---

## Dependencies

| Package | Purpose |
|--------|------|
| psycopg2-binary | PostgreSQL connection |
| pyyaml | YAML config file |
| pandas (optional) | DataFrame conversion |

Installation:
```bash
pip install psycopg2-binary pyyaml pandas
```

---

## Notes

1. **Autocommit**: PostgresManager operates in autocommit mode. If you need transactions, manage them yourself.
2. **WHERE required**: `update()` and `delete()` cannot run without a WHERE clause (a safety guard).
3. **Connection management**: Use a context manager (`with`) or call `close()` after your work.
4. **Passwords**: Do not hardcode passwords in code. Use environment variables or config files.
