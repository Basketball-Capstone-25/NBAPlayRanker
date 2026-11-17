"""Coach access to two season-long offensive profiles."""

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from application.analytics_services.team_comparison import build_team_comparison, TeamComparisonDataUnavailable
from application.api_coordination.auth_dependency import require_role
from application.api_coordination.data_quality_endpoints import DataQualityGate
from domain.baseline_recommendation import BaselineRecommender


def create_team_comparison_router(rec: BaselineRecommender, gate: DataQualityGate) -> APIRouter:
    router = APIRouter(tags=["Team profiles"])

    @router.get("/analytics/playtypes/team-comparison", dependencies=[
        Depends(require_role("recommendation")), Depends(gate),
    ])
    def team_comparison(
        response: Response,
        season: str = Query(..., min_length=7, max_length=7),
        team: str = Query(..., min_length=2, max_length=3),
        opponent: str = Query(..., min_length=2, max_length=3),
    ) -> dict:
        response.headers["Cache-Control"] = "private, no-store"
        try:
            return build_team_comparison(rec, season, team, opponent)
        except TeamComparisonDataUnavailable as error:
            raise HTTPException(503, "The source data cannot support this comparison.",
                                headers={"Cache-Control": "private, no-store"}) from error
        except ValueError as error:
            raise HTTPException(400, str(error), headers={"Cache-Control": "private, no-store"}) from error

    return router
