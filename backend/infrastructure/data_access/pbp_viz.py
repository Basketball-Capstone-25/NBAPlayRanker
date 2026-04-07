from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import HTTPException

from .pbp_constants import CANONICAL_PARQUET
from .pbp_io import read_parquet_cached


@dataclass
class HeatmapRequest:
    season: str
    team: str
    opp: str
    shot_type: Optional[str] = None
    zone: Optional[str] = None
    max_shots: int = 35000


def _load_canonical(columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Load canonical parquet via cached loader.
    """
    try:
        return read_parquet_cached(CANONICAL_PARQUET, columns=columns)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                "Canonical shots parquet not found. Run Phase 1 build first:\n"
                "  python backend/data/etl/build_pbp_pipeline.py --force"
            ),
        ) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Canonical parquet schema mismatch: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load canonical parquet: {e}") from e


def _filter_shots(df: pd.DataFrame, req: HeatmapRequest) -> pd.DataFrame:
    # canonical columns are snake_case: season, team, opp, shot_type, zone, x, y
    q = (df["season"] == req.season) & (df["team"] == req.team) & (df["opp"] == req.opp)
    if req.shot_type:
        q &= df["shot_type"] == req.shot_type
    if req.zone:
        q &= df["zone"] == req.zone
    return df.loc[q].copy()


def render_pbp_heatmap_png(
    *,
    season: str,
    team: str,
    opp: str,
    shot_type: Optional[str] = None,
    zone: Optional[str] = None,
    max_shots: int = 35000,
) -> bytes:
    """
    Render a PBP shot heatmap PNG from the canonical parquet.
    Reuses viz_shot_heatmap.render_shot_heatmap_png, adapting canonical columns.
    """
    # Lazy import so the backend can boot even if sportypy isn't installed.
    try:
        from infrastructure.visualization_and_export.viz_shot_heatmap import (
            render_shot_heatmap_png,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Shot heatmap rendering is unavailable because the visualization dependency "
                "(sportypy) could not be imported. Install it and restart the backend.\n"
                f"Import error: {e}"
            ),
        ) from e

    req = HeatmapRequest(
        season=season,
        team=team,
        opp=opp,
        shot_type=shot_type,
        zone=zone,
        max_shots=int(max_shots),
    )

    cols = ["season", "team", "opp", "shot_type", "zone", "x", "y"]
    df = _load_canonical(columns=cols)
    df = _filter_shots(df, req)

    if df.empty:
        raise HTTPException(status_code=404, detail="No shots matched these filters.")

    if len(df) > req.max_shots:
        df = df.sample(n=req.max_shots, random_state=7)

    # Adapt to the schema expected by viz_shot_heatmap.py
    shots_df = df.rename(
        columns={
            "season": "SEASON_STR",
            "team": "TEAM_ABBR",
            "opp": "OPP_ABBR",
            "shot_type": "SHOT_TYPE",
            "zone": "ZONE",
            "x": "X",
            "y": "Y",
        }
    )

    title = f"Shot Heatmap • {team} vs {opp} • {season}"
    return render_shot_heatmap_png(
        shots_df=shots_df,
        season=season,
        our_team=team,
        opp_team=opp,
        shot_type=shot_type,
        zone=zone,
        title=title,
    )


def render_pbp_heatmap_json(
    *,
    season: str,
    team: str,
    opp: str,
    shot_type: Optional[str] = None,
    zone: Optional[str] = None,
    max_shots: int = 35000,
) -> Dict[str, Any]:
    """
    Returns { ... , image_base64 } for UI usage.
    """
    try:
        from infrastructure.visualization_and_export.viz_shot_heatmap import (
            png_bytes_to_base64,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Shot heatmap rendering is unavailable because the visualization dependency "
                "(sportypy) could not be imported. Install it and restart the backend.\n"
                f"Import error: {e}"
            ),
        ) from e

    png = render_pbp_heatmap_png(
        season=season,
        team=team,
        opp=opp,
        shot_type=shot_type,
        zone=zone,
        max_shots=max_shots,
    )

    return {
        "season": season,
        "team": team,
        "opp": opp,
        "shot_type": shot_type,
        "zone": zone,
        "max_shots": int(max_shots),
        "image_base64": png_bytes_to_base64(png),
    }


def render_pbp_heatmap_base64(
    *,
    season: str,
    our: str,
    opp: str,
    shot_type: Optional[str] = None,
    zone: Optional[str] = None,
    max_shots: int = 35000,
) -> Dict[str, Any]:
    """
    Compatibility wrapper for routers that expect {caption, image_base64}.
    """
    payload = render_pbp_heatmap_json(
        season=season,
        team=our,
        opp=opp,
        shot_type=shot_type,
        zone=zone,
        max_shots=int(max_shots),
    )
    return {
        "caption": f"Shot Heatmap • {our} vs {opp} • {season}",
        "image_base64": payload["image_base64"],
    }
