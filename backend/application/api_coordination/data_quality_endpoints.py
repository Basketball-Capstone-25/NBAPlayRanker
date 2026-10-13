"""Expose the loaded Synergy source checks to the analyst workspace."""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response

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


# ---------------------------------------------------------------------------
# SCRUM-479: block or flag recommendations when the source fails its checks
# ---------------------------------------------------------------------------
QUALITY_HEADER = "X-Data-Quality"
ENFORCEMENT_ENV = "DATA_QUALITY_ENFORCEMENT"  # "block" (default) or "flag"


class DataQualityGate:
    """Run the checks once per loaded frame and guard the ranking endpoints."""

    def __init__(
        self,
        source: Callable[[], pd.DataFrame],
        seasons: Iterable[str] = EXPECTED_SEASONS,
        teams: Iterable[str] = EXPECTED_TEAMS,
    ) -> None:
        self._source = source
        self._seasons = tuple(seasons)
        self._teams = tuple(teams)
        self._frame_id: Optional[int] = None
        self._report: Optional[Dict[str, Any]] = None
        self.runs = 0

    def refresh(self) -> None:
        self._frame_id = None
        self._report = None

    def report(self) -> Dict[str, Any]:
        try:
            frame = self._source()
        except Exception as error:  # source unavailable: checks could not run
            logger.warning("Data quality gate could not read the source: %s", type(error).__name__)
            return {"status": "not_run", "safe_for_use": False, "checks": []}
        if self._report is None or self._frame_id != id(frame):
            self._report = run_data_quality_checks(
                frame, seasons=self._seasons, teams=self._teams,
            ).to_dict()
            self._frame_id = id(frame)
            self.runs += 1
        return self._report

    def status(self) -> str:
        return str(self.report().get("status", "not_run"))

    @staticmethod
    def blocking() -> bool:
        return os.getenv(ENFORCEMENT_ENV, "block").strip().lower() != "flag"

    def __call__(self) -> None:
        """FastAPI dependency: 503 when the source fails and enforcement is on."""
        report = self.report()
        if report.get("status") == "fail" and self.blocking():
            failed = [c["name"] for c in report.get("checks", []) if c.get("status") == "fail"]
            raise HTTPException(
                status_code=503,
                detail=(
                    "Recommendations are paused because the source data failed "
                    f"quality checks: {', '.join(failed)}. See /data/quality-report."
                ),
                headers={QUALITY_HEADER: "fail"},
            )


def install_data_quality_header(app: FastAPI, gate: DataQualityGate, prefixes: Iterable[str]) -> None:
    """Add X-Data-Quality to every response under the given path prefixes."""
    prefixes = tuple(prefixes)

    @app.middleware("http")
    async def _data_quality_header(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(prefixes) and QUALITY_HEADER not in response.headers:
            response.headers[QUALITY_HEADER] = gate.status()
        return response
