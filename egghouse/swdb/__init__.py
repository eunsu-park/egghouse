"""Solar / space-weather database helpers.

Domain layer on top of the generic `egghouse.database` infrastructure:

- Reference declarative schemas (`SDO_SCHEMA`, `LASCO_SCHEMA`,
  `SECCHI_SCHEMA`) — feed straight into
  `egghouse.database.create_tables_from_schema`.
- `FitsHandler` ABC + `AiaFitsHandler` — turn FITS files into validated
  metadata and DB rows.
- `register_fits_dir` — scan a directory, validate, idempotently
  upsert, optionally archive.

Only AIA ships here (what undine needs). Other instruments subclass
`FitsHandler` in their own projects.
"""

from .result import ValidationResult
from .schemas import SDO_SCHEMA, LASCO_SCHEMA, SECCHI_SCHEMA
from .handlers import FitsHandler, AiaFitsHandler, AIA_EUV_WAVELENGTHS
from .register import register_fits_dir, scan_fits, RegisterReport

__all__ = [
    "ValidationResult",
    "SDO_SCHEMA",
    "LASCO_SCHEMA",
    "SECCHI_SCHEMA",
    "FitsHandler",
    "AiaFitsHandler",
    "AIA_EUV_WAVELENGTHS",
    "register_fits_dir",
    "scan_fits",
    "RegisterReport",
]
