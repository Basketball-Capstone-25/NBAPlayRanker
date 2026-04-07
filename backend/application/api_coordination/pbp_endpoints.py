from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from application.api_coordination.auth_dependency import require_auth, require_role

from infrastructure.data_access import get_shots_csv_response, get_shots_json, render_pbp_heatmap_base64
from application.recommendation_services.pbp_shotplan import get_shotplan_json, get_shotplan_meta_options

router = APIRouter(prefix="/pbp", tags=["pbp"])

from application.api_coordination.pbp_phase2_endpoints import router as pbp_phase2_router
router.include_router(pbp_phase2_router)


@router.get("/meta/options", dependencies=[Depends(require_auth)])
def pbp_meta_options() -> Dict[str, Any]:
    """
    Lightweight meta options for Dataset2 (PBP shots) so the frontend
    can populate dropdowns without scanning the full canonical parquet.
    """
    try:
        return jsonable_encoder(get_shotplan_meta_options())
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing PBP cache files. Run Phase 1 build step. Details: {e}",
        )


@router.get("/data/shots", dependencies=[Depends(require_role("shot_analysis"))])
def pbp_data_shots(
    season: str = Query(..., description="Season like 2021-22"),
    team: str = Query(..., description="Team abbreviation (our team) like TOR"),
    opp: Optional[str] = Query(None, description="Opponent abbreviation like BOS"),
    shot_type: Optional[str] = Query(None, description="Optional shot type filter"),
    zone: Optional[str] = Query(None, description="Optional zone filter"),
    limit: int = Query(50, ge=1, le=5000),
) -> Dict[str, Any]:
    """
    JSON preview of canonical PBP shots (safe for UI tables).
    IMPORTANT: Must not return NaN/Inf (Starlette JSON compliance).
    """
    payload = get_shots_json(
        season=season,
        team=team,
        opp=opp,
        shot_type=shot_type,
        zone=zone,
        limit=int(limit),
    )
    return jsonable_encoder(payload)


# Compatibility alias: frontend pages call /pbp/shots/preview.
@router.get("/shots/preview", dependencies=[Depends(require_role("shot_analysis"))])
def pbp_shots_preview(
    season: str = Query(..., description="Season like 2021-22"),
    team: str = Query("", description="Team abbreviation (our team) like TOR"),
    opp: Optional[str] = Query(None, description="Opponent abbreviation like BOS"),
    shot_type: Optional[str] = Query(None, description="Optional shot type filter"),
    shotType: Optional[str] = Query(None, description="Optional shot type filter alias"),
    zone: Optional[str] = Query(None, description="Optional zone filter"),
    limit: int = Query(50, ge=1, le=5000),
    our: Optional[str] = Query(None, description="Alias for team"),
) -> Dict[str, Any]:
    our_team = team or our or ""
    payload = get_shots_json(
        season=season,
        team=our_team,
        opp=opp,
        shot_type=shot_type or shotType,
        zone=zone,
        limit=int(limit),
    )
    return jsonable_encoder(payload)


@router.get("/data/shots.csv", dependencies=[Depends(require_role("shot_analysis"))])
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


# Compatibility alias: frontend pages call /pbp/shots.csv.
@router.get("/shots.csv", dependencies=[Depends(require_role("shot_analysis"))])
def pbp_shots_csv(
    season: str = Query(...),
    team: str = Query(""),
    opp: Optional[str] = Query(None),
    shot_type: Optional[str] = Query(None),
    shotType: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    limit: int = Query(5000, ge=1, le=200000),
    our: Optional[str] = Query(None),
) -> StreamingResponse:
    our_team = team or our or ""
    return get_shots_csv_response(
        season=season,
        team=our_team,
        opp=opp,
        shot_type=shot_type or shotType,
        zone=zone,
        limit=int(limit),
    )


@router.get("/shotplan", dependencies=[Depends(require_role("shotplan"))])
def pbp_shotplan(
    season: str = Query(...),
    our: str = Query(..., description="Our team abbreviation"),
    opp: str = Query(..., description="Opponent team abbreviation"),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
) -> Dict[str, Any]:
    payload = get_shotplan_json(season=season, our=our, opp=opp, k=int(k), w_off=float(w_off))
    return jsonable_encoder(payload)


# Compatibility alias: some callers expect /pbp/shotplan/rank
@router.get("/shotplan/rank", dependencies=[Depends(require_role("shotplan"))])
def pbp_shotplan_rank(
    season: str = Query(...),
    our: str = Query(..., description="Our team abbreviation"),
    opp: str = Query(..., description="Opponent team abbreviation"),
    k: int = Query(5, ge=1, le=10),
    w_off: float = Query(0.7, ge=0, le=1),
) -> Dict[str, Any]:
    payload = get_shotplan_json(season=season, our=our, opp=opp, k=int(k), w_off=float(w_off))
    return jsonable_encoder(payload)


@router.get("/viz/shot-heatmap", dependencies=[Depends(require_role("viz"))])
def pbp_viz_shot_heatmap(
    season: str = Query(...),
    our: str = Query(...),
    # Alias: some callers send `team=` instead of `our=`
    team: Optional[str] = Query(None),
    opp: str = Query(...),
    shot_type: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    max_shots: int = Query(30000, ge=1000, le=250000, description="Downsample cap for plotting"),
) -> Dict[str, Any]:
    """
    Returns {caption, image_base64} like your other /viz endpoints.
    """
    our_team = our or team
    payload = render_pbp_heatmap_base64(
        season=season,
        our=our_team,
        opp=opp,
        shot_type=shot_type,
        zone=zone,
        max_shots=int(max_shots),
    )
    return jsonable_encoder(payload)


# Compatibility alias: some callers expect /pbp/viz/heatmap and send `team=`
@router.get("/viz/heatmap", dependencies=[Depends(require_role("viz"))])
def pbp_viz_heatmap(
    season: str = Query(...),
    team: str = Query(..., description="Team abbreviation (our team)"),
    opp: str = Query(...),
    shot_type: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    max_shots: int = Query(30000, ge=1000, le=250000, description="Downsample cap for plotting"),
) -> Dict[str, Any]:
    payload = render_pbp_heatmap_base64(
        season=season,
        our=team,
        opp=opp,
        shot_type=shot_type,
        zone=zone,
        max_shots=int(max_shots),
    )
    return jsonable_encoder(payload)
