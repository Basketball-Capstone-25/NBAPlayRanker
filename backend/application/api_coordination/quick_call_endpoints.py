"""Coach-only Timeout Quick-Call endpoint (SCRUM-628)."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query

from application.api_coordination.auth_dependency import require_role
from application.recommendation_services.quick_call_service import (
    QUICK_CALL_SIZE,
    QuickCallRequest,
    QuickCallService,
)

MARGIN_LIMIT = 40
OVERTIME_PERIOD = 5
REGULATION_SECONDS = 720
OVERTIME_SECONDS = 300


def validate_situation(margin: float, period: int, time_remaining: float) -> None:
    """Reject values a game clock or scoreboard cannot show."""
    if abs(margin) > MARGIN_LIMIT:
        raise HTTPException(status_code=400, detail=f"Score margin must be between -{MARGIN_LIMIT} and {MARGIN_LIMIT}.")
    limit = OVERTIME_SECONDS if period == OVERTIME_PERIOD else REGULATION_SECONDS
    if time_remaining > limit:
        raise HTTPException(status_code=400, detail=f"Time remaining cannot exceed {limit} seconds in this period.")


def create_quick_call_router(
    service: QuickCallService,
    valid_seasons: Sequence[str],
    valid_teams: Sequence[str],
    extra_dependencies: Sequence[Callable[..., Any]] = (),
) -> APIRouter:
    seasons = set(valid_seasons)
    teams = set(valid_teams)
    router = APIRouter(
        tags=["recommendation"],
        dependencies=[Depends(require_role("recommendation")), *[Depends(dep) for dep in extra_dependencies]],
    )

    @router.get("/rank-plays/quick-call")
    def rank_quick_call(
        season: str = Query(...),
        our: str = Query(..., description="Our team abbreviation."),
        opp: str = Query(..., description="Opponent team abbreviation."),
        margin: float = Query(..., allow_inf_nan=False, description="Our score minus opponent score."),
        period: int = Query(..., ge=1, le=OVERTIME_PERIOD),
        time_remaining: float = Query(..., ge=0, le=REGULATION_SECONDS, allow_inf_nan=False),
    ) -> Dict[str, Any]:
        if season not in seasons:
            raise HTTPException(status_code=400, detail=f"Unknown season '{season}'.")
        if our not in teams or opp not in teams:
            raise HTTPException(status_code=400, detail="Unknown team.")
        if our == opp:
            raise HTTPException(status_code=400, detail="Our team and opponent must be different.")
        validate_situation(margin, period, time_remaining)

        started = time.perf_counter()
        request = QuickCallRequest(season, our, opp, float(margin), int(period), float(time_remaining))
        try:
            result = service.recommend(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not result.plays:
            raise HTTPException(status_code=404, detail="No plays are available for this matchup.")

        return {
            **result.to_dict(),
            "situation": {
                "season": season,
                "our_team": our,
                "opp_team": opp,
                "margin": float(margin),
                "period": int(period),
                "time_remaining": float(time_remaining),
            },
            "k": QUICK_CALL_SIZE,
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }

    return router
