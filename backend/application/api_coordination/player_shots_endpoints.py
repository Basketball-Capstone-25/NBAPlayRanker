"""SCRUM-825: a player's made shots against the opponent, for Context ML."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from application.analytics_services.player_made_shots import (
    InvalidPlayerShotsFilter,
    PlayerShotsUnavailable,
    list_players,
    parse_seasons,
    player_made_shots,
)
from application.api_coordination.auth_dependency import require_role
from application.api_coordination.shot_data_guard import require_valid_shot_data
from domain.shot_analysis.shooter_names import normalize_athlete_id

NO_STORE = {"Cache-Control": "private, no-store"}
NAMES_CSV = Path(__file__).resolve().parents[2] / "data" / "pbp" / "shooter_names.csv"
BUILD_HINT = "Player names are not built yet. Run: python backend/data/etl/build_shooter_names.py"


@lru_cache(maxsize=2)
def _names(path: str, modified: int) -> Dict[str, str]:
    table = pd.read_csv(path, dtype={"athlete_id": str, "name": str})
    return {normalize_athlete_id(r.athlete_id): str(r.name) for r in table.itertuples() if normalize_athlete_id(r.athlete_id)}


def load_shooter_names(path: Path = NAMES_CSV) -> Dict[str, str]:
    if not path.is_file():
        raise PlayerShotsUnavailable(BUILD_HINT)
    return _names(str(path), path.stat().st_mtime_ns)


def create_player_shots_router(
    load_shots: Callable[[], pd.DataFrame],
    load_names: Callable[[], Dict[str, str]] = load_shooter_names,
) -> APIRouter:
    router = APIRouter(
        tags=["Shot analysis"],
        dependencies=[Depends(require_role("shot_tools")), Depends(require_valid_shot_data)],
    )

    def _inputs(seasons: str):
        try:
            shots, names = load_shots(), load_names()
        except PlayerShotsUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers=NO_STORE) from exc
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Shot data is not available yet.", headers=NO_STORE) from exc
        available = sorted(shots["SEASON_STR"].dropna().astype(str).unique()) if "SEASON_STR" in shots else []
        try:
            return shots, names, parse_seasons(seasons, available)
        except InvalidPlayerShotsFilter as exc:
            raise HTTPException(status_code=400, detail=str(exc), headers=NO_STORE) from exc

    def _answer(build):
        try:
            return build()
        except InvalidPlayerShotsFilter as exc:
            raise HTTPException(status_code=400, detail=str(exc), headers=NO_STORE) from exc
        except PlayerShotsUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers=NO_STORE) from exc

    @router.get("/viz/player-made-shots/seasons")
    def seasons() -> dict:
        try:
            shots = load_shots()
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="Shot data is not available yet.", headers=NO_STORE) from exc
        found = sorted(shots["SEASON_STR"].dropna().astype(str).unique()) if "SEASON_STR" in shots else []
        return {"seasons": found}

    @router.get("/viz/player-made-shots/players")
    def players(
        our: str = Query(..., min_length=2, max_length=4),
        opp: str = Query(..., min_length=2, max_length=4),
        seasons: str = Query(..., max_length=120),
    ) -> dict:
        shots, names, chosen = _inputs(seasons)
        return _answer(lambda: list_players(shots, names, our.upper(), opp.upper(), chosen))

    @router.get("/viz/player-made-shots")
    def made_shots(
        our: str = Query(..., min_length=2, max_length=4),
        opp: str = Query(..., min_length=2, max_length=4),
        seasons: str = Query(..., max_length=120),
        player: str = Query(..., min_length=1, max_length=20),
    ) -> dict:
        shots, names, chosen = _inputs(seasons)
        return _answer(lambda: player_made_shots(shots, names, our.upper(), opp.upper(), chosen, player))

    return router
