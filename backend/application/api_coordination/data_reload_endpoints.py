"""Analyst-only dataset reloads, without restarting the backend (SCRUM-490, SCRUM-491)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Response

from application.api_coordination.auth_dependency import require_role
from application.data_reload_services.pbp_reload import PbpReloadService
from application.data_reload_services.synergy_reload import (
    DataReloadError,
    ReloadInProgress,
    SynergyReloadService,
)

logger = logging.getLogger(__name__)
NO_STORE = {"Cache-Control": "no-store"}


def _run(label: str, reload: Callable[[], Dict[str, Any]], response: Response) -> Dict[str, Any]:
    response.headers.update(NO_STORE)
    try:
        result = reload()
    except ReloadInProgress as error:
        raise HTTPException(status_code=409, detail=str(error), headers=NO_STORE) from error
    except DataReloadError as error:
        logger.warning("%s reload rejected: %s", label, error)
        raise HTTPException(
            status_code=422,
            detail=f"Reload failed; the previous data is still in use. {error}",
            headers=NO_STORE,
        ) from error
    logger.info("%s reloaded: %s -> %s", label, result["previous_version"], result["dataset_version"])
    return result


def create_data_reload_router(
    synergy: Optional[SynergyReloadService],
    pbp: Optional[PbpReloadService] = None,
) -> APIRouter:
    router = APIRouter(tags=["Data reload"])

    # The request log records who called these and when (SCRUM-494).
    @router.post("/data/reload/synergy", dependencies=[Depends(require_role("data"))])
    def reload_synergy(response: Response) -> Dict[str, Any]:
        if synergy is None:
            raise HTTPException(
                status_code=503,
                detail="Play-type data did not load at startup, so it cannot be reloaded. Fix the file and restart the backend.",
                headers=NO_STORE,
            )
        return _run("Synergy", synergy.reload, response)

    @router.post("/data/reload/pbp", dependencies=[Depends(require_role("data"))])
    def reload_pbp(response: Response) -> Dict[str, Any]:
        service = pbp or PbpReloadService()
        return _run("Play-by-play", service.reload, response)

    return router
