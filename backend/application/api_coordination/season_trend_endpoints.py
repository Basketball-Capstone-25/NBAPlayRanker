"""Coach access to historical offensive play-type trends."""

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from application.analytics_services.season_trend import build_season_trend, TrendDataUnavailable
from application.api_coordination.auth_dependency import require_role
from application.api_coordination.data_quality_endpoints import DataQualityGate
from domain.baseline_recommendation import BaselineRecommender


def create_season_trend_router(rec: BaselineRecommender, gate: DataQualityGate) -> APIRouter:
    router = APIRouter(tags=["Season trends"])

    @router.get("/analytics/playtypes/season-trend", dependencies=[
        Depends(require_role("recommendation")), Depends(gate),
    ])
    def season_trend(
        response: Response,
        team: str = Query(..., min_length=2, max_length=3),
        play_type: str = Query(..., min_length=1, max_length=40),
    ) -> dict:
        response.headers["Cache-Control"] = "private, no-store"
        try:
            return build_season_trend(rec, team, play_type)
        except TrendDataUnavailable as error:
            raise HTTPException(503, "The source data cannot support this trend.",
                                headers={"Cache-Control": "private, no-store"}) from error
        except ValueError as error:
            raise HTTPException(400, str(error), headers={"Cache-Control": "private, no-store"}) from error

    return router
