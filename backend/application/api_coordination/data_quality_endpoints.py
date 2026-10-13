"""Expose the loaded Synergy source checks to the analyst workspace."""

from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any, Callable, Dict

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Response

from application.api_coordination.auth_dependency import require_role
from infrastructure.data_access.data_quality import run_data_quality_checks

logger = logging.getLogger(__name__)

# The bundled source covers these seasons and all 30 teams. Deriving the
# expectations from the source itself would hide an entirely missing team.
EXPECTED_SEASONS = tuple(f"{year}-{str(year + 1)[-2:]}" for year in range(2019, 2025))
EXPECTED_TEAMS = (
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
)


def _source_version(frame: pd.DataFrame) -> str:
    """Fingerprint the checked values and schema, without reopening a file."""
    digest = hashlib.sha256()
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype="<u8")
    digest.update(row_hashes.tobytes())
    return f"frame-sha256:{digest.hexdigest()}"


def _values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted(frame[column].dropna().astype(str).unique().tolist())


def create_data_quality_router(source: Callable[[], pd.DataFrame]) -> APIRouter:
    router = APIRouter(tags=["Data quality"])

    @router.get("/data/quality-report", dependencies=[Depends(require_role("data"))])
    def quality_report(response: Response) -> Dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        try:
            # The table and quality report use the same loaded source. A request
            # checks one snapshot, so its fingerprint and results stay together.
            frame = source().copy(deep=True)
            report = run_data_quality_checks(
                frame, seasons=EXPECTED_SEASONS, teams=EXPECTED_TEAMS,
            ).to_dict()
            report["dataset"] = {
                "name": "Synergy play-type source",
                "scope": "All source rows; independent of preview filters",
                "rows": int(len(frame)),
                "seasons": _values(frame, "SEASON"),
                "teams": _values(frame, "TEAM_ABBREVIATION"),
                "version": _source_version(frame),
            }
            report["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            return report
        except Exception as error:
            logger.warning("Data quality report unavailable: %s", type(error).__name__)
            raise HTTPException(
                status_code=503,
                detail="Data quality checks are unavailable. Please try again.",
                headers={"Cache-Control": "no-store"},
            ) from error

    return router
