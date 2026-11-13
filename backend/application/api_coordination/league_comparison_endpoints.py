"""Read-only coach access to team and league rates."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from application.analytics_services.league_comparison import build_league_comparison, LeagueDataUnavailable
from application.api_coordination.auth_dependency import require_role
from application.api_coordination.data_quality_endpoints import DataQualityGate
from domain.baseline_recommendation import BaselineRecommender


def create_league_comparison_router(rec: BaselineRecommender, gate: DataQualityGate) -> APIRouter:
    router = APIRouter(tags=["League comparison"])

    @router.get("/analytics/playtypes/league-comparison", dependencies=[
        Depends(require_role("recommendation")), Depends(gate),
    ])
    def league_comparison(
        response: Response,
        season: str = Query(..., min_length=7, max_length=7),
        team: str = Query(..., min_length=2, max_length=3),
        side: Literal["offense", "defense"] = Query("offense"),
    ) -> dict:
        response.headers["Cache-Control"] = "private, no-store"
        try:
            return build_league_comparison(rec, season, team, side)
        except LeagueDataUnavailable as error:
            raise HTTPException(503, "The source data cannot support this comparison.",
                                headers={"Cache-Control": "private, no-store"}) from error
        except ValueError as error:
            raise HTTPException(400, str(error), headers={"Cache-Control": "private, no-store"}) from error

    return router
