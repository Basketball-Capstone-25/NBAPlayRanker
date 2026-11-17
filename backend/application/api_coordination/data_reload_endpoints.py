"""Analyst-only dataset reloads, without restarting the backend (SCRUM-490)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Response

from application.api_coordination.auth_dependency import require_role
from application.data_reload_services.synergy_reload import (
    DataReloadError,
    ReloadInProgress,
    SynergyReloadService,
)

logger = logging.getLogger(__name__)
NO_STORE = {"Cache-Control": "no-store"}


def create_data_reload_router(synergy: Optional[SynergyReloadService]) -> APIRouter:
    router = APIRouter(tags=["Data reload"])

    # The request log records who called this and when (SCRUM-494).
    @router.post("/data/reload/synergy", dependencies=[Depends(require_role("data"))])
    def reload_synergy(response: Response) -> Dict[str, Any]:
        response.headers.update(NO_STORE)
        if synergy is None:
            raise HTTPException(
                status_code=503,
                detail="Play-type data did not load at startup, so it cannot be reloaded. Fix the file and restart the backend.",
                headers=NO_STORE,
            )
        try:
            result = synergy.reload()
        except ReloadInProgress as error:
            raise HTTPException(status_code=409, detail=str(error), headers=NO_STORE) from error
        except DataReloadError as error:
            logger.warning("Synergy reload rejected: %s", error)
            raise HTTPException(
                status_code=422,
                detail=f"Reload failed; the previous data is still in use. {error}",
                headers=NO_STORE,
            ) from error
        logger.info("Synergy reloaded: %s -> %s", result["previous_version"], result["dataset_version"])
        return result

    return router
