"""Season summary shared by coaches and analysts."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from application.analytics_services.season_summary import build_season_summary
from application.api_coordination.auth_dependency import require_role
from application.api_coordination.data_quality_endpoints import DataQualityGate
from domain.baseline_recommendation import BaselineRecommender
from domain.statistical_analysis.topk_uplift import UpliftDataUnavailable


def create_season_summary_router(rec: BaselineRecommender, gate: DataQualityGate) -> APIRouter:
    router = APIRouter(tags=["Season summary"])

    @router.get("/analytics/playtypes/summary", dependencies=[
        Depends(require_role("season_summary")), Depends(gate),
    ])
    def season_summary(
        response: Response,
        season: str = Query(..., min_length=7, max_length=7),
        our: str = Query(..., min_length=2, max_length=3),
        opp: str = Query(..., min_length=2, max_length=3),
        k: int = Query(5, ge=1, le=10),
        w_off: float = Query(0.7, ge=0, le=1, allow_inf_nan=False),
        period: Literal["season"] = Query("season"),
    ) -> dict:
        response.headers["Cache-Control"] = "private, no-store"
        try:
            return build_season_summary(rec, season, our, opp, k, w_off)
        except UpliftDataUnavailable as error:
            raise HTTPException(
                status_code=503, detail="The source data cannot support this summary.",
                headers={"Cache-Control": "private, no-store"},
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail=str(error),
                headers={"Cache-Control": "private, no-store"},
            ) from error

    return router
