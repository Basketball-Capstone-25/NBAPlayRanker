"""Router for PDF exports."""

from __future__ import annotations

import io
from typing import Dict, Iterable, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from application.api_coordination.auth_dependency import require_role
from domain.baseline_recommendation import BaselineRecommender, rank_playtypes_baseline
from infrastructure.visualization_and_export import build_playtype_viz_pdf


def _require_in(value: str, allowed: Set[str], label: str) -> None:
    if value not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown {label} '{value}'.")


def create_pdf_router(
    rec: BaselineRecommender,
    valid_seasons: Iterable[str],
    valid_teams: Iterable[str],
    team_names: Optional[Dict[str, str]] = None,
) -> APIRouter:
    """
    Create a FastAPI router for PDF export endpoints.
    Called by app.py: app.include_router(create_pdf_router(rec, ...))
    """
    valid_seasons_set = set(valid_seasons)
    valid_teams_set = set(valid_teams)

    router = APIRouter(tags=["export"])

    @router.get("/export/playtype-viz.pdf", dependencies=[Depends(require_role("export"))])
    def export_playtype_viz_pdf(
        season: str = Query(...),
        our: str = Query(..., description="Our team abbreviation."),
        opp: str = Query(..., description="Opponent team abbreviation."),
        play_type: str = Query(..., description="Play type (must exist in the Top-K result)."),
        k: int = Query(10, ge=1, le=10, description="Top-K used for the baseline ranking (1..10)."),
        w_off: float = Query(0.7, ge=0, le=1),
    ) -> StreamingResponse:
        """1-page PDF export with SportyPy court map + caption."""
        _require_in(season, valid_seasons_set, "season")
        _require_in(our, valid_teams_set, "our team")
        _require_in(opp, valid_teams_set, "opponent team")
        if our == opp:
            raise HTTPException(status_code=400, detail="Our team and opponent must be different.")

        w_def = float(1.0 - w_off)

        try:
            df = rank_playtypes_baseline(
                team_df=rec.team_df,
                league_df=rec.league_df,
                season=season,
                our_team=our,
                opp_team=opp,
                k=int(k),
                w_off=float(w_off),
                w_def=float(w_def),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        row = df[df["PLAY_TYPE"] == play_type]
        if row.empty:
            raise HTTPException(
                status_code=404,
                detail="Play type not found in Top-K output. Export a play type that appears in the ranking table.",
            )

        r = row.iloc[0]
        caption = (
            f"{play_type}: Pred {float(r['PPP_PRED']):.3f} PPP. "
            f"Our(off) {float(r['PPP_OFF_SHRUNK']):.3f} vs Opp(def) {float(r['PPP_DEF_SHRUNK']):.3f}."
        )

        # Render SportyPy PNG (infrastructure)
        try:
            from infrastructure.visualization_and_export.viz_sportypy import render_playtype_zone_png
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"viz_sportypy.py missing or failed to import. Error: {e}",
            )

        title = f"{our} vs {opp} • {season} • {play_type}"
        png_bytes = render_playtype_zone_png(play_type=play_type, title=title)

        # Build PDF (infrastructure)
        pdf_bytes = build_playtype_viz_pdf(
            play_type=play_type,
            season=season,
            our=our,
            opp=opp,
            caption=caption,
            png_bytes=png_bytes,
            w_off=w_off,
            w_def=w_def,
            k=k,
            team_names=team_names,
        )

        filename = f"playtype_viz_{season}_{our}_vs_{opp}_{play_type}.pdf".replace(" ", "_")
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return router
