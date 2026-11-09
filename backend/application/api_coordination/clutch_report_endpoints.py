"""SCRUM-629: clutch report endpoints for the Data Explorer (analysts only)."""

from __future__ import annotations

from typing import Callable

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from application.analytics_services.clutch_report import (
    MAX_MARGIN,
    MAX_PERIOD,
    MAX_SECONDS_LEFT,
    ClutchDataUnavailable,
    ClutchFilters,
    InvalidClutchFilter,
    build_report,
    report_csv,
    report_filename,
    situation_text,
)
from application.api_coordination.auth_dependency import require_role

NO_STORE = {"Cache-Control": "private, no-store"}


def create_clutch_report_router(load_shots: Callable[[], pd.DataFrame]) -> APIRouter:
    router = APIRouter(tags=["data"], dependencies=[Depends(require_role("data"))])

    def _report(season, team, opp, period_from, seconds_left, margin_min, margin_max) -> dict:
        try:
            filters = ClutchFilters.parse(season, team, opp, period_from, seconds_left, margin_min, margin_max)
        except InvalidClutchFilter as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            return build_report(load_shots(), filters)
        except (ClutchDataUnavailable, FileNotFoundError, OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Shot data is not available yet.") from exc

    def _params(
        season: str = Query("", max_length=7, description="Season like 2024-25."),
        team: str = Query("", max_length=4, description="Your team, e.g. TOR."),
        opp: str = Query("", max_length=4, description="Opponent, e.g. BOS."),
        period_from: int = Query(4, description=f"First period to include, 1 to {MAX_PERIOD}; overtime is always included."),
        seconds_left: int = Query(300, description=f"Seconds left in the period, 1 to {MAX_SECONDS_LEFT}."),
        margin_min: int = Query(-5, description=f"Lowest score margin for the shooting team, -{MAX_MARGIN} to {MAX_MARGIN}."),
        margin_max: int = Query(5, description="Highest score margin for the shooting team."),
    ) -> dict:
        return _report(season, team, opp, period_from, seconds_left, margin_min, margin_max)

    def _download(body, media: str, filename: str) -> Response:
        headers = {**NO_STORE, "Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=body, media_type=media, headers=headers)

    @router.get("/data/clutch-report")
    def get_clutch_report(response: Response, report: dict = Depends(_params)):
        response.headers.update(NO_STORE)
        return report

    @router.get("/data/clutch-report.csv")
    def get_clutch_report_csv(report: dict = Depends(_params)):
        return _download(report_csv(report), "text/csv; charset=utf-8", report_filename(report, "csv"))

    @router.get("/data/clutch-report.pdf")
    def get_clutch_report_pdf(report: dict = Depends(_params)):
        from infrastructure.visualization_and_export.clutch_report_pdf import render_clutch_pdf

        return _download(render_clutch_pdf(report, situation_text(report)), "application/pdf", report_filename(report, "pdf"))

    return router
