from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import hashlib
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


def _clean_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip()
    if not v or v.lower() == "all":
        return None
    return v


def _norm_code(value: Optional[str]) -> str:
    return str(value or "").strip().upper()


def _norm_text(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _stable_sample_seed(req: HeatmapRequest) -> int:
    key = "|".join(
        [
            req.season,
            req.team,
            req.opp,
            req.shot_type or "",
            req.zone or "",
            str(req.max_shots),
        ]
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


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
                "Canonical shots parquet not found. Run the Dataset2 pipeline first so "
                "shot heatmap data exists."
            ),
        ) from e
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Canonical parquet schema mismatch: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load canonical parquet: {e}") from e


def _validate_columns(df: pd.DataFrame) -> None:
    needed = ["season", "team", "opp", "shot_type", "zone", "x", "y"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=(
                "Canonical shots parquet is missing required columns for heatmap rendering: "
                + ", ".join(missing)
            ),
        )


def _build_available_options(df: pd.DataFrame, req: HeatmapRequest) -> Dict[str, Any]:
    same_season = df[df["season_norm"] == _norm_text(req.season)].copy()
    same_team = same_season[same_season["team_norm"] == _norm_code(req.team)].copy()
    same_matchup = same_team[same_team["opp_norm"] == _norm_code(req.opp)].copy()

    return {
        "seasons": sorted(df["season"].dropna().astype(str).unique().tolist())[:50],
        "teams": sorted(same_season["team"].dropna().astype(str).unique().tolist())[:50],
        "opponents_for_team": sorted(same_team["opp"].dropna().astype(str).unique().tolist())[:50],
        "shot_types_for_matchup": sorted(same_matchup["shot_type"].dropna().astype(str).unique().tolist())[:100],
        "zones_for_matchup": sorted(same_matchup["zone"].dropna().astype(str).unique().tolist())[:100],
    }


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["season"] = out["season"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip().str.upper()
    out["opp"] = out["opp"].astype(str).str.strip().str.upper()
    out["shot_type"] = out["shot_type"].astype(str).str.strip()
    out["zone"] = out["zone"].astype(str).str.strip()

    out["season_norm"] = out["season"].str.lower()
    out["team_norm"] = out["team"]
    out["opp_norm"] = out["opp"]
    out["shot_type_norm"] = out["shot_type"].str.lower()
    out["zone_norm"] = out["zone"].str.lower()

    return out


def _filter_shots(df: pd.DataFrame, req: HeatmapRequest) -> pd.DataFrame:
    season_norm = _norm_text(req.season)
    team_norm = _norm_code(req.team)
    opp_norm = _norm_code(req.opp)
    shot_type_norm = _norm_text(req.shot_type)
    zone_norm = _norm_text(req.zone)

    q = (df["season_norm"] == season_norm) & (df["team_norm"] == team_norm) & (df["opp_norm"] == opp_norm)

    if shot_type_norm:
        q &= df["shot_type_norm"] == shot_type_norm

    if zone_norm:
        q &= df["zone_norm"] == zone_norm

    return df.loc[q].copy()


def _make_caption(req: HeatmapRequest, n_total: int, n_rendered: int) -> str:
    parts = [f"Shot Heatmap • {req.team} vs {req.opp} • {req.season}"]

    if req.shot_type:
        parts.append(f"Type: {req.shot_type}")
    else:
        parts.append("Type: All")

    if req.zone:
        parts.append(f"Zone: {req.zone}")
    else:
        parts.append("Zone: All")

    parts.append(f"Matched: {n_total:,}")
    if n_rendered != n_total:
        parts.append(f"Rendered: {n_rendered:,}")

    return " • ".join(parts)


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
    try:
        from infrastructure.visualization_and_export.viz_shot_heatmap import (
            render_shot_heatmap_png,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Shot heatmap rendering is unavailable because sportypy or the heatmap "
                f"renderer could not be imported: {e}"
            ),
        ) from e

    req = HeatmapRequest(
        season=str(season).strip(),
        team=_norm_code(team),
        opp=_norm_code(opp),
        shot_type=_clean_optional(shot_type),
        zone=_clean_optional(zone),
        max_shots=int(max_shots),
    )

    cols = ["season", "team", "opp", "shot_type", "zone", "x", "y"]
    raw = _load_canonical(columns=cols)
    _validate_columns(raw)

    df = _prepare_df(raw)
    matched = _filter_shots(df, req)

    if matched.empty:
        options = _build_available_options(df, req)
        raise HTTPException(
            status_code=404,
            detail={
                "message": "No shots matched these heatmap filters.",
                "requested": {
                    "season": req.season,
                    "team": req.team,
                    "opp": req.opp,
                    "shot_type": req.shot_type,
                    "zone": req.zone,
                },
                "available": options,
            },
        )

    render_df = matched
    if len(render_df) > req.max_shots:
        render_df = render_df.sample(n=req.max_shots, random_state=_stable_sample_seed(req))

    shots_df = render_df.rename(
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

    title = _make_caption(req, n_total=len(matched), n_rendered=len(render_df))

    return render_shot_heatmap_png(
        shots_df=shots_df,
        season=req.season,
        our_team=req.team,
        opp_team=req.opp,
        shot_type=req.shot_type,
        zone=req.zone,
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
    Returns a UI-friendly payload including the base64 image and matched counts.
    """
    try:
        from infrastructure.visualization_and_export.viz_shot_heatmap import (
            png_bytes_to_base64,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Shot heatmap rendering is unavailable because sportypy or the heatmap "
                f"renderer could not be imported: {e}"
            ),
        ) from e

    req = HeatmapRequest(
        season=str(season).strip(),
        team=_norm_code(team),
        opp=_norm_code(opp),
        shot_type=_clean_optional(shot_type),
        zone=_clean_optional(zone),
        max_shots=int(max_shots),
    )

    cols = ["season", "team", "opp", "shot_type", "zone", "x", "y"]
    raw = _load_canonical(columns=cols)
    _validate_columns(raw)

    df = _prepare_df(raw)
    matched = _filter_shots(df, req)

    if matched.empty:
        options = _build_available_options(df, req)
        raise HTTPException(
            status_code=404,
            detail={
                "message": "No shots matched these heatmap filters.",
                "requested": {
                    "season": req.season,
                    "team": req.team,
                    "opp": req.opp,
                    "shot_type": req.shot_type,
                    "zone": req.zone,
                },
                "available": options,
            },
        )

    render_df = matched
    if len(render_df) > req.max_shots:
        render_df = render_df.sample(n=req.max_shots, random_state=_stable_sample_seed(req))

    png = render_pbp_heatmap_png(
        season=req.season,
        team=req.team,
        opp=req.opp,
        shot_type=req.shot_type,
        zone=req.zone,
        max_shots=req.max_shots,
    )

    return {
        "season": req.season,
        "team": req.team,
        "opp": req.opp,
        "shot_type": req.shot_type,
        "zone": req.zone,
        "max_shots": int(req.max_shots),
        "n_shots_total": int(len(matched)),
        "n_shots_rendered": int(len(render_df)),
        "caption": _make_caption(req, n_total=len(matched), n_rendered=len(render_df)),
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
    Compatibility wrapper for routers that expect a base64 payload.
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
        "season": payload["season"],
        "team": payload["team"],
        "opp": payload["opp"],
        "shot_type": payload["shot_type"],
        "zone": payload["zone"],
        "max_shots": payload["max_shots"],
        "n_shots": payload["n_shots_rendered"],
        "n_shots_total": payload["n_shots_total"],
        "n_shots_rendered": payload["n_shots_rendered"],
        "caption": payload["caption"],
        "image_base64": payload["image_base64"],
    }