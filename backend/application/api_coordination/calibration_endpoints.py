"""Analyst-only calibration reporting; separate from existing model metrics."""

from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from application.api_coordination.auth_dependency import require_role
from domain.statistical_analysis.calibration import compute_calibration
from infrastructure.model_management.ml_models import load_offense_dataset


def create_calibration_router(team_df: pd.DataFrame, league_df: pd.DataFrame) -> APIRouter:
    router = APIRouter(tags=["Calibration"])

    @lru_cache(maxsize=16)
    def report(n_splits: int, n_bins: int):
        data = load_offense_dataset(team_df, league_df)
        return compute_calibration(data, n_splits=n_splits, n_bins=n_bins)

    @router.get("/metrics/calibration", dependencies=[Depends(require_role("analytics"))])
    def calibration(n_splits: int = Query(5, ge=1, le=10), n_bins: int = Query(10, ge=2, le=20)):
        try:
            return report(n_splits, n_bins)
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return router
