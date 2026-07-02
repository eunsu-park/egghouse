"""NOAA SWPC real-time JSON parsers for space-weather ingestion.

Parses the rolling-window products served under services.swpc.noaa.gov into
DataFrames whose columns match the ``rt_*`` / ``swpc_*`` space-weather tables.
The live JSON time field ``time_tag`` maps to the table column ``datetime``.

These parsers are pure (pandas only — no DB, no network), so they do not need
the ``egghouse[database]`` extras. Promoted from the ``solaris-data`` project's
``core/swpc.py`` so every SOLARIS sub-project shares one tested implementation
instead of re-deriving the schemas. Endpoint schemas verified live 2026-06-26
(see the SOLARIS vault plan).
"""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

__all__ = [
    "parse_xray",
    "parse_proton",
    "parse_solar_wind",
    "parse_kp_1m",
    "parse_kp_forecast",
    "parse_solar_probabilities",
    "parse_alerts",
    "parse_3day_forecast",
]


def _to_utc_naive(series) -> pd.Series:
    """Parse SWPC time tags to tz-naive UTC.

    SWPC mixes 'Z'-suffixed (tz-aware) and bare (naive) time tags across
    products; both are UTC. Forcing tz-naive UTC keeps them consistent and
    avoids psycopg2 shifting tz-aware values to the server's local zone when
    written to a `TIMESTAMP` (without time zone) column.
    """
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def parse_xray(data: list) -> pd.DataFrame:
    """Parse GOES X-ray flux JSON into rt_goes_xray rows.

    The feed has two records per timestamp distinguished by `energy`
    ("0.05-0.4nm" short band, "0.1-0.8nm" long band); they are pivoted to
    one row per (satellite, datetime) with short/long flux columns.
    """
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
    short = (df[df['energy'].str.startswith('0.05')]
             [['time_tag', 'satellite', 'flux']]
             .rename(columns={'flux': 'xrs_short_w_m2'}))
    long = (df[df['energy'].str.startswith('0.1')]
            [['time_tag', 'satellite', 'flux']]
            .rename(columns={'flux': 'xrs_long_w_m2'}))
    out = short.merge(long, on=['time_tag', 'satellite'], how='outer')
    out['datetime'] = _to_utc_naive(out['time_tag'])
    return out[['satellite', 'datetime', 'xrs_short_w_m2', 'xrs_long_w_m2']]


# Header column name -> rt table column, per solar-wind product.
_SW_PLASMA_MAP = {
    'density': 'density_p_cc',
    'speed': 'speed_km_s',
    'temperature': 'temperature_k',
}
_SW_MAG_MAP = {
    'bx_gsm': 'bx_gsm_nt',
    'by_gsm': 'by_gsm_nt',
    'bz_gsm': 'bz_gsm_nt',
    'lon_gsm': 'lon_gsm_deg',
    'lat_gsm': 'lat_gsm_deg',
    'bt': 'bt_nt',
}


def parse_solar_wind(data: list, kind: str, source: str = 'DSCOVR') -> pd.DataFrame:
    """Parse a SWPC header-row-plus-rows solar-wind product.

    Args:
        data: ``[[header...], [row...], ...]`` as served by /products/solar-wind/.
        kind: 'plasma' or 'mag' (selects the column mapping / target table).
        source: L1 monitor label stored in the `source` column.
    """
    if not data or len(data) < 2:
        return pd.DataFrame()
    header, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=header)
    colmap = _SW_PLASMA_MAP if kind == 'plasma' else _SW_MAG_MAP
    out = pd.DataFrame()
    out['datetime'] = _to_utc_naive(df['time_tag'])
    for src, dst in colmap.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors='coerce')
    out['source'] = source
    return out


def parse_kp_1m(data: list) -> pd.DataFrame:
    """Parse the estimated 1-min planetary K index into rt_kp rows."""
    df = pd.DataFrame(data)
    if df.empty:
        return df
    out = pd.DataFrame()
    out['datetime'] = _to_utc_naive(df['time_tag'])
    out['estimated_kp'] = pd.to_numeric(df.get('estimated_kp'), errors='coerce')
    out['kp'] = pd.to_numeric(df.get('kp'), errors='coerce')
    return out


def parse_proton(data: list) -> pd.DataFrame:
    """Parse GOES integral proton flux JSON into rt_goes_proton rows.

    Long format: one row per (satellite, time, energy threshold), e.g.
    `>=10 MeV` (which feeds the NOAA S-scale).
    """
    df = pd.DataFrame(data)
    if df.empty:
        return df
    out = pd.DataFrame()
    out['satellite'] = df['satellite']
    out['datetime'] = _to_utc_naive(df['time_tag'])
    out['energy'] = df['energy']
    out['flux'] = pd.to_numeric(df['flux'], errors='coerce')
    return out


def parse_kp_forecast(data: list) -> pd.DataFrame:
    """Parse the 3-hourly Kp forecast into rt_kp_forecast rows.

    Header-row product ['time_tag','kp','observed','noaa_scale']; `observed` is
    observed/estimated/predicted, separating history from the forecast tail.
    """
    if not data or len(data) < 2:
        return pd.DataFrame()
    header, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=header)
    out = pd.DataFrame()
    out['datetime'] = _to_utc_naive(df['time_tag'])
    out['kp'] = pd.to_numeric(df['kp'], errors='coerce')
    out['observed_flag'] = df.get('observed')
    scale = df.get('noaa_scale')
    out['noaa_scale'] = scale.where(scale != 'null') if scale is not None else None
    return out


def parse_solar_probabilities(data: list) -> pd.DataFrame:
    """Parse C/M/X flare + 10 MeV proton probabilities into swpc_solar_probabilities."""
    df = pd.DataFrame(data)
    if df.empty:
        return df
    out = pd.DataFrame()
    out['valid_date'] = pd.to_datetime(df['date']).dt.date
    for cls in ('c', 'm', 'x'):
        for d in (1, 2, 3):
            out[f'{cls}_class_{d}_day'] = pd.to_numeric(
                df.get(f'{cls}_class_{d}_day'), errors='coerce')
    for d in (1, 2, 3):
        out[f'proton_10mev_{d}_day'] = pd.to_numeric(
            df.get(f'10mev_protons_{d}_day'), errors='coerce')
    out['polar_cap_absorption'] = df.get('polar_cap_absorption')
    return out


def parse_alerts(data: list) -> pd.DataFrame:
    """Parse the SWPC alerts/watches/warnings feed into swpc_alerts rows."""
    df = pd.DataFrame(data)
    if df.empty:
        return df
    out = pd.DataFrame()
    out['product_id'] = df['product_id']
    out['issue_datetime'] = _to_utc_naive(df['issue_datetime'])
    out['message'] = df['message']
    return out.dropna(subset=['product_id', 'issue_datetime'])


def parse_3day_forecast(text: str) -> pd.DataFrame:
    """Parse the 3-day forecast text into one swpc_3day_forecast row (raw + issue time).

    Issue stamp form: 'Issued 2026 Jun 26 0030 UTC'. Returns empty if not found.
    """
    m = re.search(r'Issued:?\s+(\d{4})\s+([A-Za-z]{3})\s+(\d{1,2})\s+(\d{4})\s+UTC', text)
    if not m:
        return pd.DataFrame()
    issued = datetime.strptime(
        f"{m.group(1)} {m.group(2)} {int(m.group(3)):02d} {m.group(4)}", "%Y %b %d %H%M")
    return pd.DataFrame([{'issued_at': issued, 'raw_text': text}])
