# backend/pbp_shots.py

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from fastapi.responses import StreamingResponse

from pbp_loader import get_pbp_canonical_df


# ---------------------------------------------------------------------
# Paths + caching
# ---------------------------------------------------------------------
# We keep Dataset2 assets under:
#   backend/data/pbp/
# and cache under:
#   backend/data/pbp/cache/
#
# This file loads the canonical parquet once per backend process,
# then serves filters quickly without re-reading from disk on every request.

_BACKEND_DIR = Path(__file__).resolve().parent
_CACHE_DIR = _BACKEND_DIR / "data" / "pbp" / "cache"

# This is the canonical parquet produced by your pipeline step.
# It should contain 1 row per shot event with standardized columns.
CANONICAL_PARQUET = _CACHE_DIR / "shots_canonical.parquet"

# module-level cache (load once per process)
_SHOTS_DF: Optional[pd.DataFrame] = None


def _load_canonical_shots_df() -> pd.DataFrame:
    """
    Load canonical shots once and normalize columns to the explorer-friendly schema.

    Important behavior:
    - If the canonical parquet is missing, we try to BUILD it through pbp_loader
      from the raw PBP parquet / clean parquet instead of failing immediately.
    - We normalize both snake_case canonical columns and legacy uppercase columns
      into one stable uppercase schema the frontend already expects.
    """
    global _SHOTS_DF
    if _SHOTS_DF is not None:
        return _SHOTS_DF

    # pbp_loader already handles: ensure canonical cache exists, reuse process cache,
    # and rebuild when the upstream fingerprint changes.
    df = get_pbp_canonical_df(ensure=True, force_rebuild=False).copy()

    # Normalize expected columns for downstream code (preview + CSV export).
    rename_map: Dict[str, str] = {}

    # Canonical snake_case -> stable explorer schema
    snake_to_ui = {
        "season": "SEASON_STR",
        "team": "TEAM_ABBR",
        "opp": "OPP_ABBR",
        "game_id": "GAME_ID",
        "period": "PERIOD",
        "clock_sec": "CLOCK_SEC",
        "shot_type": "SHOT_TYPE",
        "zone": "ZONE",
        "is_make": "IS_MAKE",
        "points": "POINTS",
        "x": "X",
        "y": "Y",
        "dist": "DIST",
        "angle": "ANGLE",
    }
    for src, dst in snake_to_ui.items():
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst

    # Legacy alternate names -> stable explorer schema
    legacy_to_ui = {
        "SEASON": "SEASON_STR",
        "TEAM": "TEAM_ABBR",
        "OPP_TEAM": "OPP_ABBR",
        "opponent": "OPP_ABBR",
        "SHOTTYPE": "SHOT_TYPE",
        "SHOT_ZONE": "ZONE",
        "LOC_X": "X",
        "loc_x": "X",
        "LOC_Y": "Y",
        "loc_y": "Y",
        "MADE": "IS_MAKE",
        "PTS": "POINTS",
    }
    for src, dst in legacy_to_ui.items():
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst

    if rename_map:
        df = df.rename(columns=rename_map)

    _SHOTS_DF = df
    return df


# ---------------------------------------------------------------------
# JSON safety: replace NaN/Inf recursively
# ---------------------------------------------------------------------

def _sanitize_json(obj: Any) -> Any:
    """
    FastAPI/JSON cannot encode NaN or Infinity.
    This removes them from nested lists/dicts so the response never explodes.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


def _season_sort_key(s: str) -> int:
    # "2025-26" -> 2025, "2021-22" -> 2021
    try:
        return int(str(s).split("-")[0])
    except Exception:
        return -1


def get_meta_options() -> Dict[str, Any]:
    """
    Returns dropdown options for Dataset2 shots explorer:
      { seasons, teams, shotTypes, zones }
    """
    df = _load_canonical_shots_df()

    seasons = []
    if "SEASON_STR" in df.columns:
        seasons = sorted(
            [x for x in df["SEASON_STR"].dropna().unique().tolist()],
            key=_season_sort_key,
        )

    teams = []
    if "TEAM_ABBR" in df.columns:
        teams = sorted([x for x in df["TEAM_ABBR"].dropna().unique().tolist()])

    shot_types = []
    if "SHOT_TYPE" in df.columns:
        shot_types = sorted([x for x in df["SHOT_TYPE"].dropna().unique().tolist()])

    zones = []
    if "ZONE" in df.columns:
        zones = sorted([x for x in df["ZONE"].dropna().unique().tolist()])

    return {
        "seasons": seasons,
        "teams": teams,
        "shotTypes": shot_types,
        "zones": zones,
        "metadata": {"source": str(CANONICAL_PARQUET.name)},
    }


def _filter_df(
    df: pd.DataFrame,
    *,
    season: str,
    team: str,
    opp: Optional[str],
    shot_type: Optional[str],
    zone: Optional[str],
) -> pd.DataFrame:
    out = df

    if "SEASON_STR" in out.columns:
        out = out[out["SEASON_STR"] == season]

    if "TEAM_ABBR" in out.columns:
        out = out[out["TEAM_ABBR"] == team]

    if opp and "OPP_ABBR" in out.columns:
        out = out[out["OPP_ABBR"] == opp]

    if shot_type and "SHOT_TYPE" in out.columns:
        out = out[out["SHOT_TYPE"] == shot_type]

    if zone and "ZONE" in out.columns:
        out = out[out["ZONE"] == zone]

    return out


def _select_output_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer a consistent "canonical preview schema" so the table looks stable.
    If some columns don't exist, we just return whatever is present.
    """
    preferred = [
        "SEASON_STR",
        "TEAM_ABBR",
        "OPP_ABBR",
        "GAME_ID",
        "PERIOD",
        "CLOCK_SEC",
        "SHOT_TYPE",
        "ZONE",
        "IS_MAKE",
        "POINTS",
        "X",
        "Y",
    ]
    cols = [c for c in preferred if c in df.columns]
    return df[cols] if cols else df


def get_shots_json(
    *,
    season: str,
    team: str,
    opp: Optional[str],
    shot_type: Optional[str],
    zone: Optional[str],
    limit: int,
) -> Dict[str, Any]:
    """
    JSON preview for the Shots Explorer page.

    IMPORTANT: frontend expects:
      { columns: string[], rows: object[] }
    """
    df = _load_canonical_shots_df()

    filtered = _filter_df(
        df, season=season, team=team, opp=opp, shot_type=shot_type, zone=zone
    )
    total = int(filtered.shape[0])

    out_df = _select_output_cols(filtered).head(int(limit)).copy()

    # Replace NaN/Inf at the dataframe level first
    out_df = out_df.replace([np.inf, -np.inf], np.nan)

    columns = list(out_df.columns)
    rows = out_df.to_dict(orient="records")
    rows = _sanitize_json(rows)

    return {
        "season": season,
        "team": team,
        "opp": opp,
        "shot_type": shot_type,
        "zone": zone,
        "total_rows": total,
        "returned_rows": len(rows),
        "columns": columns,
        "rows": rows,
        "metadata": {"source": str(CANONICAL_PARQUET.name)},
    }


def get_shots_csv_response(
    *,
    season: str,
    team: str,
    opp: Optional[str],
    shot_type: Optional[str],
    zone: Optional[str],
    limit: int,
) -> StreamingResponse:
    """
    CSV export for Shots Explorer.
    """
    df = _load_canonical_shots_df()
    filtered = _filter_df(
        df, season=season, team=team, opp=opp, shot_type=shot_type, zone=zone
    )
    out_df = _select_output_cols(filtered).head(int(limit)).copy()

    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)

    filename = f"pbp_shots_{season}_{team}" + (f"_vs_{opp}" if opp else "") + ".csv"
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )