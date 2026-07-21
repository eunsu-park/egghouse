"""CME-catalogue parsers for space-weather ingestion.

Two independent coronal-mass-ejection sources, parsed into DataFrames whose
columns match the ``donki_cme`` / ``cactus_cme`` space-weather tables:

- ``parse_donki_cme`` — NASA CCMC **DONKI** CME feed (``api.nasa.gov/DONKI/CME``),
  a JSON list of analyst-curated CMEs with WSA-ENLIL analysis (speed, width,
  source region, Earth-directed flag, estimated shock arrival, predicted Kp).
  The operational primary source.
- ``parse_cactus`` — SIDC/ROB **CACTus** automated LASCO catalogue
  (``cmecat.txt``), a fixed ``|``-delimited list of coronagraph detections
  (onset, principal angle, angular width, velocity). An independent,
  fully-automated cross-check (no human supervision — use with caution).

Like ``swpc.py`` these are pure (pandas only — no DB, no network), so they do
not need the ``egghouse[database]`` extras. DONKI CME schema and CACTus
``cmecat.txt`` format verified live 2026-07-22.
"""

from __future__ import annotations

import pandas as pd

from .swpc import _to_utc_naive

__all__ = ["parse_donki_cme", "parse_cactus"]


def _best_analysis(analyses: list) -> dict | None:
    """The CME's most-accurate WSA-ENLIL analysis (latest as fallback)."""
    if not analyses:
        return None
    for a in analyses:
        if a.get("isMostAccurate"):
            return a
    return analyses[-1]


def parse_donki_cme(data: list) -> pd.DataFrame:
    """Parse the NASA DONKI CME feed into ``donki_cme`` rows.

    One row per CME activity, keyed by ``activity_id``. Measurement fields
    (speed, half-angle, lat/lon, type, 21.5 R_s time) come from the CME's
    most-accurate analysis; the Earth-directed flag, estimated shock arrival,
    and predicted Kp are pulled from that analysis' WSA-ENLIL runs. A CME with
    no analysis yet keeps NULL measurements (still recorded so it is not lost).

    Args:
        data: The JSON list returned by ``/DONKI/CME``.
    """
    if not data:
        return pd.DataFrame()
    rows = []
    for c in data:
        best = _best_analysis(c.get("cmeAnalyses") or [])
        row = {
            "activity_id": c.get("activityID"),
            "catalog": c.get("catalog"),
            "start_time": c.get("startTime"),
            "source_location": c.get("sourceLocation") or None,
            "active_region": c.get("activeRegionNum"),
            "instruments": ", ".join(
                i.get("displayName", "") for i in (c.get("instruments") or [])
            ) or None,
            "note": c.get("note") or None,
            "link": c.get("link"),
            "speed": None, "half_angle": None, "latitude": None,
            "longitude": None, "type": None, "time21_5": None,
            "is_earth_directed": False,
            "enlil_shock_arrival": None, "enlil_kp": None,
        }
        if best:
            row.update({
                "speed": best.get("speed"),
                "half_angle": best.get("halfAngle"),
                "latitude": best.get("latitude"),
                "longitude": best.get("longitude"),
                "type": best.get("type"),
                "time21_5": best.get("time21_5"),
            })
            arrivals, kps, earth = [], [], False
            for e in best.get("enlilList") or []:
                if e.get("estimatedShockArrivalTime"):
                    arrivals.append(e["estimatedShockArrivalTime"])
                kps += [e[k] for k in ("kp_90", "kp_135", "kp_180")
                        if e.get(k) is not None]
                earth = (earth or bool(e.get("isEarthGB"))
                         or bool(e.get("isEarthMinorImpact")))
            # A predicted Earth shock arrival or Kp is the robust Earth-directed
            # signal (isEarthGB alone misses direct hits reported via impactList).
            row["is_earth_directed"] = earth or bool(arrivals) or bool(kps)
            row["enlil_shock_arrival"] = min(arrivals) if arrivals else None
            row["enlil_kp"] = max(kps) if kps else None
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["activity_id", "start_time"])
    for col in ("start_time", "time21_5", "enlil_shock_arrival"):
        df[col] = _to_utc_naive(df[col])
    for col in ("speed", "half_angle", "latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["active_region"] = pd.to_numeric(
        df["active_region"], errors="coerce").astype("Int64")
    df["enlil_kp"] = pd.to_numeric(df["enlil_kp"], errors="coerce").astype("Int64")
    return df


def parse_cactus(text: str) -> pd.DataFrame:
    """Parse a CACTus LASCO CME catalogue (``cmecat.txt``) into ``cactus_cme`` rows.

    Fixed ``|``-delimited columns:
    ``id | t0 (onset) | dt0 (liftoff h) | pa (principal angle, deg) |
    da (angular width, deg) | v (median, km/s) | dv | minv | maxv | halo``.
    The rolling real-time id is not stable across the catalogue lifetime, so
    rows are keyed downstream by (t0, pa). A ``halo`` flag (e.g. ``II``/``IV``)
    marks (partial-)halo detections; blank otherwise.

    Args:
        text: Raw ``cmecat.txt`` body.
    """
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 9:
            continue
        cid = parts[0].strip("[] ")
        if not cid.isdigit():
            continue
        try:
            rows.append({
                "cactus_id": cid,
                "t0": parts[1].replace("/", "-"),
                "dt0": int(parts[2]), "pa": int(parts[3]), "da": int(parts[4]),
                "v": int(parts[5]), "dv": int(parts[6]),
                "minv": int(parts[7]), "maxv": int(parts[8]),
                "halo": (parts[9].strip() if len(parts) > 9 else "") or None,
            })
        except ValueError:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["t0"] = _to_utc_naive(df["t0"])
    return df
