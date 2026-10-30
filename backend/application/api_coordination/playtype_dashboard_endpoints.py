"""Coach-only historical matchup analytics; no request-time model fitting."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from application.api_coordination.auth_dependency import require_role
from application.api_coordination.data_quality_endpoints import DataQualityGate
from domain.baseline_recommendation import BaselineRecommender
from domain.baseline_recommendation.playtype_dashboard import PlaytypeDashboardService


def create_playtype_dashboard_router(rec: BaselineRecommender, data_quality_gate: DataQualityGate) -> APIRouter:
    router = APIRouter(tags=["Matchup dashboard"])
    service = PlaytypeDashboardService(rec.team_df, rec.league_df, rec.raw_df.attrs.get("dataset_version", "unavailable"))

    @router.get("/analytics/playtypes", dependencies=[Depends(require_role("recommendation")), Depends(data_quality_gate)])
    def playtype_dashboard(
        response: Response,
        season: str = Query(..., min_length=1, max_length=16),
        our: str = Query(..., min_length=1, max_length=8),
        opp: str = Query(..., min_length=1, max_length=8),
        min_poss: int = Query(20, ge=1, le=10000),
    ) -> dict[str, Any]:
        response.headers["Cache-Control"] = "private, no-store"
        try:
            return service.build(season, our, opp, min_poss)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error), headers={"Cache-Control": "private, no-store"}) from error

    return router
