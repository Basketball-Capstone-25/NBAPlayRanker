# backend/pbp_endpoints.py

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from pbp_shots import get_meta_options, get_shots_csv_response, get_shots_json
from pbp_shotplan import get_shotplan_json, get_shotplan_meta_options
from pbp_viz import render_pbp_heatmap_base64

router = APIRouter(prefix="/pbp", tags=["pbp"])

from pbp_phase2_endpoints import router as pbp_phase2_router
router.include_router(pbp_phase2_router)


@router.get("/meta/options")
def pbp_meta_options() -> Dict[str, Any]:
    """
    Dataset2 meta options for:
    - Shot Explorer
    - Shot Heatmap
    - Shot Plan

    Prefer canonical shot meta first because it includes shotTypes + zones
    from the actual filtered Dataset2 shot rows.
    Fallback to shotplan aggregate meta if canonical preview metadata is unavailable.
    """
    try:
        return jsonable_encoder(get_meta_options())
    except FileNotFoundError:
        try:
            return jsonable_encoder(get_shotplan_meta_options())
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Missing PBP cache files. Run the Dataset2 build step first. Details: {e}",
            )


@router.get("/data/shots")
def pbp_data_shots(
    season: str = Query(..., description="Season like 2024-25"),
    team: str = Query(..., description="Team abbreviation like TOR"),
    opp: Optional[str] = Query(None, description="Opponent abbreviation like BOS"),
    shot_type: Optional[str] = Query(None, description="Optional shot type filter"),
    zone: Optional[str] = Query(None, description="Optional zone filter"),
    limit: int = Query(50, ge=1, le=5000),
) -> Dict[str, Any]:
    payload = get_shots_json(
        season=season,
        team=team,
        opp=opp,
        shot_type=shot_type,
        zone=zone,
        limit=int(limit),
    )
    return jsonable_encoder(payload)


@router.get("/data/shots.csv")
def pbp_data_shots_csv(
    season: str = Query(...),
    team: str = Query(...),
    opp: Optional[str] = Query(None),
    shot_type: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=200000),
) -> StreamingResponse:
    return get_shots_csv_response(
        season=season,
        team=team,
        opp=opp,
        shot_type=shot_type,
        zone=zone,
        limit=int(limit),
    )


@router.get("/shots/preview")
def pbp_shots_preview(
    season: str = Query(..., description="Season like 2024-25"),
    team: str = Query("", description="Team abbreviation like TOR"),
    our: Optional[str] = Query(None, description="Alias for team"),
    opp: Optional[str] = Query(None, description="Opponent abbreviation like BOS"),
    shot_type: Optional[str] = Query(None, description="Optional shot type filter"),
    shotType: Optional[str] = Query(None, description="Alias for shot_type"),
    zone: Optional[str] = Query(None, description="Optional zone filter"),
    limit: int = Query(50, ge=1, le=5000),
) -> Dict[str, Any]:
    """
    Compatibility route used by the Shot Explorer frontend.
    """
    team_value = (team or our or "").strip().upper()
    if not team_value:
        raise HTTPException(status_code=422, detail="Missing required team/our parameter.")

    payload = get_shots_json(
        season=season,
        team=team_value,
        opp=(opp.strip().upper() if opp else None),
        shot_type=shot_type or shotType,
        zone=zone,
        limit=int(limit),
    )
    return jsonable_encoder(payload)


@router.get("/shots.csv")
def pbp_shots_csv(
    season: str = Query(...),
    team: str = Query(""),
    our: Optional[str] = Query(None),
    opp: Optional[str] = Query(None),
    shot_type: Optional[str] = Query(None),
    shotType: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=200000),
) -> StreamingResponse:
    """
    Compatibility CSV route used by the Shot Explorer frontend.
    """
    team_value = (team or our or "").strip().upper()
    if not team_value:
        raise HTTPException(status_code=422, detail="Missing required team/our parameter.")

    return get_shots_csv_response(
        season=season,
        team=team_value,
        opp=(opp.strip().upper() if opp else None),
        shot_type=shot_type or shotType,
        zone=zone,
        limit=int(limit),
    )


@router.get("/shotplan")
def pbp_shotplan(
    season: str = Query(...),
    our: str = Query(..., description="Our team abbreviation"),
    opp: str = Query(..., description="Opponent team abbreviation"),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
) -> Dict[str, Any]:
    payload = get_shotplan_json(
        season=season,
        our=our.strip().upper(),
        opp=opp.strip().upper(),
        k=int(k),
        w_off=float(w_off),
    )
    return jsonable_encoder(payload)


@router.get("/shotplan/rank")
def pbp_shotplan_rank(
    season: str = Query(...),
    our: str = Query(..., description="Our team abbreviation"),
    opp: str = Query(..., description="Opponent team abbreviation"),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
) -> Dict[str, Any]:
    """
    Compatibility alias expected by the Shot Plan frontend.
    """
    payload = get_shotplan_json(
        season=season,
        our=our.strip().upper(),
        opp=opp.strip().upper(),
        k=int(k),
        w_off=float(w_off),
    )
    return jsonable_encoder(payload)


@router.get("/viz/shot-heatmap")
def pbp_viz_shot_heatmap(
    season: str = Query(...),
    our: Optional[str] = Query(None, description="Our team abbreviation"),
    team: Optional[str] = Query(None, description="Alias for our"),
    opp: str = Query(...),
    shot_type: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    max_shots: int = Query(30000, ge=1000, le=250000, description="Downsample cap for plotting"),
) -> Dict[str, Any]:
    """
    Returns a dynamic matchup-specific heatmap payload:
    {
      season, team, opp, shot_type, zone, max_shots,
      n_shots, n_shots_total, n_shots_rendered,
      caption, image_base64
    }
    """
    our_team = (our or team or "").strip().upper()
    if not our_team:
        raise HTTPException(status_code=422, detail="Missing required our/team parameter.")

    payload = render_pbp_heatmap_base64(
        season=season,
        our=our_team,
        opp=opp.strip().upper(),
        shot_type=shot_type,
        zone=zone,
        max_shots=int(max_shots),
    )
    return jsonable_encoder(payload)


@router.get("/viz/heatmap")
def pbp_viz_heatmap(
    season: str = Query(...),
    team: str = Query(..., description="Team abbreviation (our team)"),
    opp: str = Query(...),
    shot_type: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    max_shots: int = Query(30000, ge=1000, le=250000, description="Downsample cap for plotting"),
) -> Dict[str, Any]:
    """
    Legacy alias for callers that still use /pbp/viz/heatmap?team=...
    """
    payload = render_pbp_heatmap_base64(
        season=season,
        our=team.strip().upper(),
        opp=opp.strip().upper(),
        shot_type=shot_type,
        zone=zone,
        max_shots=int(max_shots),
    )
    return jsonable_encoder(payload)