"""Solar / space-weather database helpers.

Domain layer on top of the generic `egghouse.database` infrastructure:

- Reference declarative schemas (`SDO_SCHEMA`, `LASCO_SCHEMA`,
  `SECCHI_SCHEMA`) — feed straight into
  `egghouse.database.create_tables_from_schema`.
- `FitsHandler` ABC + `AiaFitsHandler` — turn FITS files into validated
  metadata and DB rows.
- `register_fits_dir` — scan a directory, validate, idempotently
  upsert, optionally archive.
- `get_sdo_best_match` / `get_sdo_best_matches` — read side: resolve the
  stored ``sdo`` record(s) closest to target time(s).
- `parse_*` (swpc) — pure NOAA SWPC real-time JSON parsers (pandas only, no
  DB); ``rt_*`` / ``swpc_*`` space-weather table rows.

Only AIA ships here (what undine needs). Other instruments subclass
`FitsHandler` in their own projects.
"""

from .result import ValidationResult
from .schemas import SDO_SCHEMA, LASCO_SCHEMA, SECCHI_SCHEMA
from .handlers import FitsHandler, AiaFitsHandler, AIA_EUV_WAVELENGTHS
from .swpc import (
    parse_xray,
    parse_proton,
    parse_solar_wind,
    parse_kp_1m,
    parse_kp_forecast,
    parse_solar_probabilities,
    parse_alerts,
    parse_3day_forecast,
)

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
    "get_sdo_best_match",
    "get_sdo_best_matches",
    "parse_xray",
    "parse_proton",
    "parse_solar_wind",
    "parse_kp_1m",
    "parse_kp_forecast",
    "parse_solar_probabilities",
    "parse_alerts",
    "parse_3day_forecast",
]

# `register`/`query` pull in `egghouse.database` (→ psycopg2). Load them
# lazily so the pure NOAA SWPC parsers (`egghouse.swdb.swpc`) — and this
# package's schemas/handlers — import without the `[database]` extra, per
# egghouse's guard-heavy-optional-deps convention. Consumers that use the
# DB side (`get_sdo_best_match`, `register_fits_dir`, …) resolve unchanged
# on first attribute access; psycopg2 is only required then.
_LAZY = {
    "register_fits_dir": "register",
    "scan_fits": "register",
    "RegisterReport": "register",
    "get_sdo_best_match": "query",
    "get_sdo_best_matches": "query",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
