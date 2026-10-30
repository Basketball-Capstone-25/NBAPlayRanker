"""SCRUM-627: Data Explorer endpoints for the play-by-play shot dataset (analysts only)."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from application.analytics_services.shot_explorer import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    InvalidShotFilter,
    ShotDataUnavailable,
    ShotExplorerService,
    ShotFilters,
)
from application.api_coordination.auth_dependency import require_role

NO_STORE = {"Cache-Control": "private, no-store"}


def create_shot_explorer_router(service: ShotExplorerService) -> APIRouter:
    router = APIRouter(tags=["data"], dependencies=[Depends(require_role("data"))])

    def _filters(season, team, opp, shot_type, zone, result, page=1, page_size=DEFAULT_PAGE_SIZE) -> ShotFilters:
        try:
            return ShotFilters.parse(season, team, opp, shot_type, zone, result, page, page_size)
        except InvalidShotFilter as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _export(fmt: str, **kwargs) -> Response:
        filters = _filters(**kwargs)
        try:
            export = service.export(filters)
        except ShotDataUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if fmt == "csv":
            body, media = export.to_csv(), "text/csv; charset=utf-8"
        else:
            body, media = json.dumps(export.to_json(), indent=2), "application/json"
        headers = {**NO_STORE, "Content-Disposition": f'attachment; filename="{export.filename(fmt)}"'}
        return Response(content=body, media_type=media, headers=headers)

    @router.get("/data/pbp-shots")
    def get_shots(
        season: str = Query("", max_length=7, description="Season like 2024-25."),
        team: str = Query("", max_length=4, description="Shooting team, e.g. TOR."),
        opp: Optional[str] = Query(None, max_length=4),
        shot_type: Optional[str] = Query(None, max_length=40),
        zone: Optional[str] = Query(None, max_length=40),
        result: Optional[str] = Query(None, max_length=10, description="made or missed"),
        page: int = Query(1),
        page_size: int = Query(DEFAULT_PAGE_SIZE, description=f"1 to {MAX_PAGE_SIZE}"),
    ):
        filters = _filters(season, team, opp, shot_type, zone, result, page, page_size)
        try:
            return service.page(filters)
        except ShotDataUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.get("/data/pbp-shots.csv")
    def export_shots_csv(
        season: str = Query("", max_length=7),
        team: str = Query("", max_length=4),
        opp: Optional[str] = Query(None, max_length=4),
        shot_type: Optional[str] = Query(None, max_length=40),
        zone: Optional[str] = Query(None, max_length=40),
        result: Optional[str] = Query(None, max_length=10),
    ):
        return _export("csv", season=season, team=team, opp=opp, shot_type=shot_type, zone=zone, result=result)

    @router.get("/data/pbp-shots.json")
    def export_shots_json(
        season: str = Query("", max_length=7),
        team: str = Query("", max_length=4),
        opp: Optional[str] = Query(None, max_length=4),
        shot_type: Optional[str] = Query(None, max_length=40),
        zone: Optional[str] = Query(None, max_length=40),
        result: Optional[str] = Query(None, max_length=10),
    ):
        return _export("json", season=season, team=team, opp=opp, shot_type=shot_type, zone=zone, result=result)

    return router


def default_shot_loader():
    """Canonical play-by-play shots, loaded once per process by the data-access layer."""
    from infrastructure.data_access.pbp_shots import load_canonical_shots

    return load_canonical_shots()
