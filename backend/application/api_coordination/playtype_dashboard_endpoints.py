"""Coach-only historical matchup analytics; no request-time model fitting."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from application.api_coordination.auth_dependency import require_role
from application.api_coordination.data_quality_endpoints import DataQualityGate
from domain.baseline_recommendation import BaselineRecommender
from domain.baseline_recommendation.playtype_dashboard import PlaytypeDashboardService


def create_playtype_dashboard_router(rec: BaselineRecommender, data_quality_gate: DataQualityGate) -> APIRouter:
    """Keep one dashboard service for the currently loaded play-type tables."""
    router = APIRouter(tags=["Matchup dashboard"])
    built: dict[str, Any] = {}

    def service() -> PlaytypeDashboardService:
        """Rebuild the dashboard service when a reload replaces the source frame."""
        # SCRUM-490: rebuild only when an analyst reload swapped the tables.
        if built.get("frame") is not rec.raw_df:
            built["service"] = PlaytypeDashboardService(
                rec.team_df, rec.league_df, rec.raw_df.attrs.get("dataset_version", "unavailable"),
            )
            built["frame"] = rec.raw_df
        return built["service"]

    @router.get("/analytics/playtypes", dependencies=[Depends(require_role("recommendation")), Depends(data_quality_gate)])
    def playtype_dashboard(
        response: Response,
        season: str = Query(..., min_length=1, max_length=16),
        our: str = Query(..., min_length=1, max_length=8),
        opp: str = Query(..., min_length=1, max_length=8),
        min_poss: int = Query(20, ge=1, le=10000),
    ) -> dict[str, Any]:
        """Compare offensive usage and efficiency with the opponent’s defensive profile and flag small samples."""
        response.headers["Cache-Control"] = "private, no-store"
        try:
            return service().build(season, our, opp, min_poss)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error), headers={"Cache-Control": "private, no-store"}) from error

    return router
