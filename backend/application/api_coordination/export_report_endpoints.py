"""Generic /export/report endpoint built on the unified export contract (SCRUM-485)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from application.access_control_services.access_control_service import check_user_access
from application.api_coordination.auth_dependency import require_auth
from application.api_coordination.export_contract import (
    ExportContract,
    ExportFormatUnavailable,
    ExportPayload,
)
from domain.baseline_recommendation import BaselineRecommender, rank_playtypes_baseline
from domain.statistical_analysis.topk_uplift import UpliftDataUnavailable, compute_topk_uplift
from infrastructure.visualization_and_export.table_pdf import render_table_pdf
from application.api_coordination.report_snapshot import (
    MAX_SNAPSHOT_BYTES, SNAPSHOT_ROLES, validate_snapshot,
)

# Each report keeps the role check of the page endpoint that shows the same data.
REPORT_ROLES: Dict[str, str] = {
    "baseline_rankings": "recommendation",
    "team_playtypes": "data",
    "topk_uplift": "analytics",
}

TEAM_PLAYTYPE_COLUMNS = [
    "SEASON", "TEAM_ABBREVIATION", "TEAM_NAME", "SIDE", "PLAY_TYPE", "GP", "POSS",
    "POSS_PCT", "PPP", "EFG_PCT", "SCORE_POSS_PCT", "TOV_POSS_PCT", "RELIABILITY_WEIGHT",
]


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.replace({np.nan: None}).to_dict(orient="records")


def create_export_report_router(
    rec: BaselineRecommender,
    valid_seasons: Iterable[str],
    valid_teams: Iterable[str],
    source_path: Path,
) -> APIRouter:
    seasons = set(valid_seasons)
    teams = set(valid_teams)
    with Path(source_path).open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()

    contract = ExportContract()
    for report in set(REPORT_ROLES) | set(SNAPSHOT_ROLES):
        contract.register_pdf_renderer(report, render_table_pdf)

    router = APIRouter(tags=["export"])

    def _season(season: Optional[str]) -> str:
        if season not in seasons:
            raise HTTPException(status_code=400, detail=f"Unknown season '{season}'.")
        return season

    def _matchup(our: Optional[str], opp: Optional[str]) -> None:
        if our not in teams or opp not in teams:
            raise HTTPException(status_code=400, detail="our and opp must be known team codes.")
        if our == opp:
            raise HTTPException(status_code=400, detail="Our team and opponent must be different.")

    def _baseline(season: str, our: str, opp: str, k: int, w_off: float) -> pd.DataFrame:
        try:
            return rank_playtypes_baseline(
                team_df=rec.team_df, league_df=rec.league_df, season=season,
                our_team=our, opp_team=opp, k=k, w_off=w_off, w_def=1.0 - w_off,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def build_payload(report: str, season: str, our: Optional[str], opp: Optional[str],
                      side: Optional[str], k: int, w_off: float) -> ExportPayload:
        if report == "baseline_rankings":
            _matchup(our, opp)
            df = _baseline(season, our, opp, k, w_off)
            return ExportPayload(
                report=report,
                filters={"season": season, "our": our, "opp": opp, "k": k, "w_off": w_off},
                columns=list(df.columns), rows=_records(df),
                metadata={"title": f"Baseline Top-{k}: {our} vs {opp}, {season}"},
            )
        if report == "team_playtypes":
            if side not in (None, "offense", "defense"):
                raise HTTPException(status_code=400, detail="side must be 'offense' or 'defense'.")
            df = rec.team_df[rec.team_df["SEASON"] == season]
            if side:
                df = df[df["SIDE"] == side]
            columns = [c for c in TEAM_PLAYTYPE_COLUMNS if c in df.columns]
            df = df[columns].sort_values(["TEAM_ABBREVIATION", "SIDE", "PLAY_TYPE"])
            if df.empty:
                raise HTTPException(status_code=404, detail="No rows match these filters.")
            return ExportPayload(
                report=report, filters={"season": season, "side": side},
                columns=columns, rows=_records(df),
                metadata={"title": f"Team play types, {season}"},
            )
        # topk_uplift
        _matchup(our, opp)
        ranked = _baseline(season, our, opp, k, w_off)
        try:
            result = compute_topk_uplift(
                rec.team_df, ranked, season=season, our_team=our, opp_team=opp,
                k=k, w_off=w_off,
                provenance={"source_file": Path(source_path).name, "source_sha256": digest},
            )
        except UpliftDataUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        rows = result["rankings"]
        return ExportPayload(
            report=report,
            filters={"season": season, "our": our, "opp": opp, "k": k, "w_off": w_off},
            columns=list(rows[0]) if rows else [], rows=rows,
            metadata={
                "title": f"Top-{k} PPP uplift: {our} vs {opp}, {season}",
                "team_season_ppp": result["team_season"]["ppp"],
                "topk_modeled_ppp": result["topk"]["modeled_ppp"],
                "uplift_ppp": result["uplift_ppp"],
                "limitations": result["limitations"],
            },
        )

    @router.get("/export/report")
    def export_report(
        report: str = Query(..., description="baseline_rankings, team_playtypes or topk_uplift"),
        format: str = Query(..., description="csv, json or pdf"),
        season: Optional[str] = Query(None),
        our: Optional[str] = Query(None),
        opp: Optional[str] = Query(None),
        side: Optional[str] = Query(None),
        k: int = Query(5, ge=1, le=10),
        w_off: float = Query(0.7, ge=0, le=1, allow_inf_nan=False),
        user: dict = Depends(require_auth),
    ) -> Response:
        role = REPORT_ROLES.get(report)
        if role is None:
            raise HTTPException(status_code=400, detail=f"Unknown report '{report}'. Use one of: {', '.join(REPORT_ROLES)}.")
        if not check_user_access(user.get("role"), role):
            raise HTTPException(status_code=403, detail=f"Role '{user.get('role')}' cannot export '{report}'.")
        payload = build_payload(report, _season(season), our, opp, side, k, w_off)
        try:
            file = contract.build_export(payload, format)
        except ExportFormatUnavailable as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=file.content, media_type=file.media_type, headers=file.headers)

    @router.post("/export/report")
    async def export_displayed_report(request: Request, user: dict = Depends(require_auth)) -> Response:
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            raise HTTPException(status_code=415, detail="Send the report as JSON.")
        chunks = bytearray()
        async for chunk in request.stream():
            if len(chunks) + len(chunk) > MAX_SNAPSHOT_BYTES:
                raise HTTPException(status_code=413, detail="The report is too large.")
            chunks.extend(chunk)
        try:
            payload, fmt = validate_snapshot(json.loads(chunks))
        except (ValueError, TypeError, RecursionError, OverflowError) as exc:
            raise HTTPException(status_code=400, detail="Invalid report: " + str(exc)[:180]) from exc
        if not check_user_access(user.get("role"), SNAPSHOT_ROLES[payload.report]):
            raise HTTPException(status_code=403, detail="Your account cannot export this report.")
        file = await run_in_threadpool(contract.build_export, payload, fmt)
        return Response(content=file.content, media_type=file.media_type, headers={
            **file.headers,
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        })

    return router
