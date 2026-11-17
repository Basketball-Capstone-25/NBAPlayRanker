"""Coach access to independent, source-checked possession rates."""

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from application.analytics_services.possession_outcomes import build_possession_outcomes, OutcomeDataUnavailable
from application.api_coordination.auth_dependency import require_role
from application.api_coordination.data_quality_endpoints import DataQualityGate
from domain.baseline_recommendation import BaselineRecommender


def create_possession_outcomes_router(rec: BaselineRecommender, gate: DataQualityGate) -> APIRouter:
    router = APIRouter(tags=["Possession outcomes"])

    @router.get("/analytics/playtypes/possession-outcomes", dependencies=[
        Depends(require_role("recommendation")), Depends(gate),
    ])
    def possession_outcomes(
        response: Response,
        season: str = Query(..., min_length=7, max_length=7),
        team: str = Query(..., min_length=2, max_length=3),
    ) -> dict:
        response.headers["Cache-Control"] = "private, no-store"
        try:
            return build_possession_outcomes(rec, season, team)
        except OutcomeDataUnavailable as error:
            raise HTTPException(503, "The source data cannot support these possession rates.",
                                headers={"Cache-Control": "private, no-store"}) from error
        except ValueError as error:
            raise HTTPException(400, str(error), headers={"Cache-Control": "private, no-store"}) from error

    return router
