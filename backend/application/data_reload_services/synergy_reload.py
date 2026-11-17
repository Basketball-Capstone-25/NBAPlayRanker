"""Reload the Synergy play-type source while the backend keeps running (SCRUM-490).

The file is read and checked in full before anything changes. If any step
fails, the tables already in memory stay in use and the error says why.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import pandas as pd

from domain.baseline_recommendation import BaselineRecommender
from infrastructure.data_access.data_quality import run_data_quality_checks
from infrastructure.data_access.dataset_metadata import read_versioned_csv
from infrastructure.model_management.ml_models import validate_prediction_provenance


class DataReloadError(Exception):
    """The new data was rejected; the previous data is still loaded."""


class ReloadInProgress(Exception):
    """Another reload of the same dataset is still running."""


class SynergyReloadService:
    def __init__(
        self,
        rec: BaselineRecommender,
        predictions_path: Path,
        seasons: Iterable[str],
        teams: Iterable[str],
        on_reloaded: Callable[[BaselineRecommender, Optional[pd.DataFrame]], None],
        loader: Callable[[str], BaselineRecommender] = BaselineRecommender,
    ) -> None:
        self._rec = rec
        self._predictions_path = Path(predictions_path)
        self._seasons = tuple(seasons)
        self._teams = tuple(teams)
        self._on_reloaded = on_reloaded
        self._loader = loader
        self._lock = threading.Lock()

    def reload(self) -> Dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise ReloadInProgress("A Synergy reload is already running. Try again shortly.")
        try:
            return self._reload()
        finally:
            self._lock.release()

    def _reload(self) -> Dict[str, Any]:
        path = Path(self._rec.source_path)
        previous = self._rec.raw_df.attrs.get("dataset_version", "unavailable")

        try:
            candidate = self._loader(str(path))
        except FileNotFoundError as error:
            raise DataReloadError(f"{path.name} was not found.") from error
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            raise DataReloadError(f"{path.name} could not be read ({type(error).__name__}).") from error
        if candidate.team_df.empty:
            raise DataReloadError(f"{path.name} has no regular-season play-type rows.")

        report = run_data_quality_checks(candidate.raw_df, seasons=self._seasons, teams=self._teams).to_dict()
        if report.get("status") != "pass":
            failed = [c["name"] for c in report.get("checks", []) if c.get("status") != "pass"]
            raise DataReloadError(f"{path.name} failed its data checks: {', '.join(failed) or 'checks could not run'}.")

        # Context ML joins the saved predictions to these tables, so they must
        # come from the same source. Rebuild them offline first if it changed.
        predictions: Optional[pd.DataFrame] = None
        if self._predictions_path.exists():
            try:
                predictions = read_versioned_csv(self._predictions_path)
                validate_prediction_provenance(predictions, candidate.team_df)
            except (OSError, ValueError) as error:
                raise DataReloadError(
                    f"Saved play-type predictions do not match {path.name}. Rebuild them offline, then reload."
                ) from error

        self._rec.adopt(candidate)
        self._on_reloaded(self._rec, predictions)
        current = self._rec.raw_df.attrs.get("dataset_version", "unavailable")
        return {
            "dataset": "synergy",
            "status": "reloaded",
            "source_file": path.name,
            "previous_version": previous,
            "dataset_version": current,
            "changed": current != previous,
            "rows": int(len(self._rec.raw_df)),
            "seasons": sorted(self._rec.team_df["SEASON"].dropna().astype(str).unique().tolist()),
            "quality_status": report.get("status"),
        }
