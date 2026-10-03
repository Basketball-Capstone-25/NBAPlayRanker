"""Analyst-only Top-K uplift evidence and equivalent CSV/JSON downloads."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from application.api_coordination.auth_dependency import require_role
from domain.baseline_recommendation import BaselineRecommender, rank_playtypes_baseline
from domain.statistical_analysis.topk_uplift import compute_topk_uplift, UpliftDataUnavailable


def _csv_evidence(payload: dict) -> str:
    """Repeat complete summary/provenance on each independently useful CSV row."""
    summary = {key: value for key, value in payload.items() if key != "rankings"}
    flat = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                flat[f"{key}_{child_key}"] = child_value
        elif isinstance(value, list):
            flat[key] = json.dumps(value, ensure_ascii=False)
        else:
            flat[key] = value
    rows = [{**flat, **row} for row in payload["rankings"]]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def create_topk_uplift_router(rec: BaselineRecommender, source_path: Path) -> APIRouter:
    """Reuse the loaded baseline tables; fingerprint their source at startup."""
    source_path = Path(source_path)
    with source_path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    provenance = {
        "source_file": source_path.name,
        "source_sha256": digest,
        "aggregation": "player rows aggregated by season, team, play type and side",
        "baseline_points_source": "PTS (no fallback to rounded PPP)",
        "recommendation_source": "/rank-plays/baseline with identical filters",
    }
    router = APIRouter(tags=["metrics"], dependencies=[Depends(require_role("analytics"))])

    def evidence(
        season: str = Query(...),
        our: str = Query(..., description="Our team abbreviation."),
        opp: str = Query(..., description="Opponent team abbreviation."),
        k: int = Query(5, ge=1, le=10),
        w_off: float = Query(0.7, ge=0, le=1, allow_inf_nan=False),
    ) -> dict:
        if our == opp:
            raise HTTPException(status_code=400, detail="Our team and opponent must be different.")
        try:
            ranked = rank_playtypes_baseline(
                rec.team_df, rec.league_df, season, our, opp,
                k=k, w_off=w_off, w_def=1.0 - w_off,
            )
            return compute_topk_uplift(
                rec.team_df, ranked, season=season, our_team=our, opp_team=opp,
                k=k, w_off=w_off, provenance=provenance,
            )
        except UpliftDataUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            status = 404 if str(exc).startswith("No data for this matchup") else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    @router.get("/metrics/topk-uplift")
    def topk_uplift(payload: dict = Depends(evidence)) -> dict:
        """Return a reproducible descriptive PPP comparison, not a causal claim."""
        return payload

    def filename(payload: dict, extension: str) -> str:
        return (
            f"topk_uplift_{payload['season']}_{payload['our_team']}"
            f"_vs_{payload['opp_team']}_top{payload['k']}.{extension}"
        )

    @router.get("/metrics/topk-uplift.json")
    def topk_uplift_json(payload: dict = Depends(evidence)) -> JSONResponse:
        return JSONResponse(
            content=payload,
            headers={"Content-Disposition": f'attachment; filename="{filename(payload, "json")}"'},
        )

    @router.get("/metrics/topk-uplift.csv")
    def topk_uplift_csv(payload: dict = Depends(evidence)) -> StreamingResponse:
        return StreamingResponse(
            iter([_csv_evidence(payload)]), media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename(payload, "csv")}"'},
        )

    return router
