"""Read queries over the solar-image database.

Read side of `egghouse.swdb`, complementing the write side in
`register.py`. These helpers resolve the stored record(s) closest to one
or more target times for a given telescope/channel against the shared
``sdo`` table. They depend only on `egghouse.database`.

Promoted from the `solaris-data` project so that every SOLARIS
sub-project (data acquisition there, modelling in `undine`) reads the
shared corpus through one tested implementation instead of re-deriving
the SQL.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from egghouse.database import PostgresManager


def get_sdo_best_match(
    db_config: dict,
    telescope: str,
    channel: str,
    target_time: datetime,
    time_range_minutes: int = 30,
    require_quality_zero: bool = True,
) -> Optional[dict]:
    """Get the SDO record closest to ``target_time`` within a window.

    Args:
        db_config: Database configuration dict (``PostgresManager`` kwargs).
        telescope: Telescope name ('aia' or 'hmi').
        channel: Channel name as stored in the ``sdo`` table, e.g. '193'.
        target_time: Target datetime to find the closest match for.
        time_range_minutes: Half-width of the search window, in minutes.
        require_quality_zero: If True, only consider ``quality = 0`` rows.

    Returns:
        The matching row as a dict (all ``sdo`` columns, incl. ``file_path``),
        or None if no row falls within the window.
    """
    quality_condition = "AND quality = 0" if require_quality_zero else ""

    sql = f"""
        SELECT * FROM sdo
        WHERE telescope = %s
          AND channel = %s
          {quality_condition}
          AND datetime BETWEEN %s AND %s
        ORDER BY ABS(EXTRACT(EPOCH FROM (datetime - %s)))
        LIMIT 1
    """

    start_time = target_time - timedelta(minutes=time_range_minutes)
    end_time = target_time + timedelta(minutes=time_range_minutes)

    with PostgresManager(**db_config) as db:
        result = db.execute(
            sql,
            (telescope, channel, start_time, end_time, target_time),
            fetch=True,
        )

        if result:
            return result[0]
        return None


def get_sdo_best_matches(
    db_config: dict,
    telescope: str,
    channel: str,
    target_times: list[datetime],
    time_range_minutes: int = 30,
    require_quality_zero: bool = True,
) -> pd.DataFrame:
    """Get the SDO records closest to several target times.

    Calls :func:`get_sdo_best_match` once per target time and assembles
    the hits into a DataFrame. Target times with no match are dropped; an
    extra ``target_time`` column records which request each row answers.

    Args:
        db_config: Database configuration dict (``PostgresManager`` kwargs).
        telescope: Telescope name ('aia' or 'hmi').
        channel: Channel name as stored in the ``sdo`` table, e.g. '193'.
        target_times: Target datetimes to find closest matches for.
        time_range_minutes: Half-width of the search window, in minutes.
        require_quality_zero: If True, only consider ``quality = 0`` rows.

    Returns:
        A DataFrame of best matches (empty if none were found).
    """
    results = []
    for target_time in target_times:
        match = get_sdo_best_match(
            db_config,
            telescope,
            channel,
            target_time,
            time_range_minutes,
            require_quality_zero,
        )
        if match:
            match["target_time"] = target_time
            results.append(match)

    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()
