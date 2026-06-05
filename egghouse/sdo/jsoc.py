"""JSOC export utilities for SDO data acquisition (drms-based).

This module wraps the `drms` Python client (https://github.com/sunpy/drms)
so callers can express common SDO/AIA acquisition tasks in one line:

- :func:`jsoc_export` submits a JSOC export request and returns the URL
  list of the resulting files. Pair it with the egghouse.transfer
  downloaders for retry/atomic-write semantics.
- :func:`aia_euv_query` composes a record-set string selecting AIA EUV
  records near a given list of timestamps, optionally filtered to a
  specific channel set.
- :func:`cached_correction_table` and :func:`cached_pointing_table` cache
  the (slow-to-fetch) aiapy calibration tables on disk so batch jobs do
  not refetch them on every record.

Network-dependent third-party modules (`drms`, `aiapy`, `astropy`) are
imported inside the functions that need them, so simply importing this
module does not require any of those dependencies. Tests for the local
query composer therefore run with no network access.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence, TYPE_CHECKING, Union

from .dem_response import AIA_DEM_WAVELENGTHS

if TYPE_CHECKING:  # only for type hints, never imported at runtime
    import drms
    from astropy.table import Table


# DRMS time format used in record-set strings (e.g. "2014.01.01_12:00:00").
_DRMS_TIME_FMT = "%Y.%m.%d_%H:%M:%S"

# Default series name for AIA EUV level-1 records.
AIA_LEV1_EUV_SERIES = "aia.lev1_euv_12s"


def jsoc_export(
    query: str,
    *,
    email: str,
    method: str = "url",
    protocol: str = "fits",
    client: Optional["drms.Client"] = None,
) -> list[str]:
    """Submit a JSOC export request and return the resulting URL list.

    Parameters
    ----------
    query : str
        DRMS record-set string, e.g. produced by :func:`aia_euv_query`.
    email : str
        JSOC export email (must be registered at jsoc.stanford.edu).
    method : str, default ``"url"``
        Export method; ``"url"`` blocks server-side until the dataset is
        staged and returns concrete URLs.
    protocol : str, default ``"fits"``
        File protocol for staged files.
    client : drms.Client, optional
        Reusable client. When ``None`` a fresh one is constructed from
        ``email``.

    Returns
    -------
    list[str]
        URLs of staged files, in the order JSOC returned them. May be
        empty when no records match the query.

    Raises
    ------
    RuntimeError
        If the export request did not finish successfully.
    """
    import drms

    if client is None:
        client = drms.Client(email=email)
    request = client.export(query, method=method, protocol=protocol)
    request.wait()
    if not request.has_succeeded():
        raise RuntimeError(
            f"JSOC export did not succeed: status={request.status}, "
            f"query={query!r}"
        )
    return list(request.urls.url)


def aia_euv_query(
    times: Sequence[datetime],
    *,
    wavelengths: Sequence[int] = AIA_DEM_WAVELENGTHS,
    series: str = AIA_LEV1_EUV_SERIES,
    tolerance: timedelta = timedelta(seconds=12),
) -> str:
    """Compose a DRMS record-set string selecting AIA EUV records.

    For each timestamp in ``times`` the query selects records within
    ``tolerance`` starting at that time. All channels in ``wavelengths``
    are kept; everything else in the series is filtered out via a record
    predicate. Multiple timestamps are concatenated into a single export
    request.

    Example
    -------
    >>> from datetime import datetime, timedelta
    >>> q = aia_euv_query([datetime(2014, 1, 1, 12, 0, 0)])
    >>> q.startswith("aia.lev1_euv_12s[2014.01.01_12:00:00_TAI/")
    True

    Parameters
    ----------
    times : sequence of datetime
        Timestamps to select. Must be non-empty. Naive datetimes are
        interpreted as TAI (which is what JSOC expects).
    wavelengths : sequence of int, default the six DEM channels
        AIA EUV passbands in Angstroms to retain.
    series : str, default ``aia.lev1_euv_12s``
        DRMS series name.
    tolerance : timedelta, default 12 s
        How far past each timestamp to scan for a matching record.

    Returns
    -------
    str
        DRMS record-set string ready for :func:`jsoc_export`.
    """
    if not times:
        raise ValueError("`times` must be non-empty")
    if not wavelengths:
        raise ValueError("`wavelengths` must be non-empty")
    step_seconds = int(tolerance.total_seconds())
    if step_seconds <= 0:
        raise ValueError(f"tolerance must be positive; got {tolerance!r}")

    time_clauses = "".join(
        f"[{t.strftime(_DRMS_TIME_FMT)}_TAI/{step_seconds}s]" for t in times
    )
    wave_predicate = " OR ".join(f"WAVELNTH={int(w)}" for w in wavelengths)
    return f"{series}{time_clauses}[? {wave_predicate} ?]"


def cached_correction_table(path: Union[str, os.PathLike]) -> "Table":
    """Load (or fetch and cache) the aiapy degradation correction table.

    The first call fetches the table from JSOC via
    `aiapy.calibrate.util.get_correction_table` and pickles it to
    ``path``. Subsequent calls deserialize from disk and skip the
    network round-trip.

    Parameters
    ----------
    path : path-like
        Pickle cache location. Parent directories are created on demand.

    Returns
    -------
    astropy.table.Table
        The correction table.
    """
    path = Path(path)
    if path.is_file():
        with open(path, "rb") as f:
            return pickle.load(f)
    from aiapy.calibrate.utils import get_correction_table

    table = get_correction_table()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(table, f)
    return table


def cached_pointing_table(
    path: Union[str, os.PathLike],
    *,
    start: datetime,
    end: datetime,
) -> "Table":
    """Load (or fetch and cache) the aiapy pointing table for [start, end).

    Like :func:`cached_correction_table`, this exists to avoid repeatedly
    fetching the same time-bound table from JSOC across many records.
    The ``start`` and ``end`` are only used when the cache is being
    populated; a stale cache file is reused as-is. Callers needing a
    fresh window should delete the file first.

    Parameters
    ----------
    path : path-like
        Pickle cache location.
    start, end : datetime
        Time range to request from aiapy. ``end`` is exclusive in the
        usual aiapy convention.

    Returns
    -------
    astropy.table.Table
        The pointing table.
    """
    path = Path(path)
    if path.is_file():
        with open(path, "rb") as f:
            return pickle.load(f)
    from aiapy.calibrate.util import get_pointing_table
    from astropy.time import Time

    table = get_pointing_table(Time(start), Time(end))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(table, f)
    return table
