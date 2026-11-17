"""Reload the play-by-play shot data while the backend keeps running (SCRUM-491).

The cleaned shots file is read, converted and checked in memory first. Only a
file that passes is written as the new canonical table and swapped in; any
failure leaves the shots already in memory in use and says why.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from application.data_reload_services.synergy_reload import DataReloadError, ReloadInProgress
from infrastructure.data_access import pbp_loader
from infrastructure.data_access.dataset_metadata import read_versioned_parquet
from infrastructure.data_access.pbp_cache import build_meta, fingerprint_file, write_json_atomic, write_parquet_atomic
from infrastructure.data_access.pbp_clean import CANONICAL_COLUMNS, build_canonical_from_clean
from infrastructure.data_access.pbp_constants import (
    CANONICAL_META_JSON,
    CANONICAL_PARQUET,
    CANONICAL_SCHEMA_VERSION,
    CLEAN_PARQUET,
)

# Without these a shot cannot be placed in a game, team or on the court.
REQUIRED_CLEAN_COLUMNS = ("SEASON_STR", "TEAM_ABBR", "OPP_ABBR", "GAME_ID", "X", "Y", "POINTS")
_SEASON = re.compile(r"^\d{4}-\d{2}$")


class PbpReloadService:
    def __init__(
        self,
        clean_path: Path = CLEAN_PARQUET,
        canonical_path: Path = CANONICAL_PARQUET,
        meta_path: Path = CANONICAL_META_JSON,
        on_reloaded: Callable[[], None] = lambda: None,
    ) -> None:
        self._clean_path = Path(clean_path)
        self._canonical_path = Path(canonical_path)
        self._meta_path = Path(meta_path)
        self._on_reloaded = on_reloaded
        self._lock = threading.Lock()

    def reload(self) -> Dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise ReloadInProgress("A play-by-play reload is already running. Try again shortly.")
        try:
            return self._reload()
        finally:
            self._lock.release()

    def _reload(self) -> Dict[str, Any]:
        name = self._clean_path.name
        current: Optional[pd.DataFrame] = pbp_loader.loaded_pbp_canonical()
        previous = current.attrs.get("dataset_version", "unavailable") if current is not None else "not loaded"

        try:
            clean = read_versioned_parquet(self._clean_path)
        except FileNotFoundError as error:
            raise DataReloadError(f"{name} was not found.") from error
        except (OSError, ValueError) as error:
            raise DataReloadError(f"{name} could not be read ({type(error).__name__}).") from error

        missing = [column for column in REQUIRED_CLEAN_COLUMNS if column not in clean.columns]
        if missing:
            raise DataReloadError(f"{name} is missing columns: {', '.join(missing)}.")
        try:
            canonical = build_canonical_from_clean(clean)
        except (KeyError, ValueError, TypeError) as error:
            raise DataReloadError(f"{name} could not be converted ({type(error).__name__}).") from error
        if canonical.empty:
            raise DataReloadError(f"{name} has no usable shots (each needs a season, team, game and court location).")
        bad_seasons = sorted({s for s in canonical["season"].unique() if not _SEASON.match(str(s))})
        if bad_seasons:
            raise DataReloadError(f"{name} has unrecognised seasons: {', '.join(map(str, bad_seasons[:5]))}.")

        try:
            write_parquet_atomic(canonical, self._canonical_path)
            write_json_atomic(self._meta_path, build_meta(
                fingerprint=fingerprint_file(self._clean_path, schema_version=CANONICAL_SCHEMA_VERSION),
                extra={"rows": int(len(canonical)), "columns": CANONICAL_COLUMNS,
                       "notes": "Rebuilt by an analyst reload (SCRUM-491)."},
            ))
            loaded = read_versioned_parquet(self._canonical_path)
        except (OSError, ValueError) as error:
            raise DataReloadError(f"The checked shots could not be saved ({type(error).__name__}).") from error

        pbp_loader.swap_pbp_canonical(loaded)
        self._on_reloaded()
        version = loaded.attrs.get("dataset_version", "unavailable")
        return {
            "dataset": "pbp_shots",
            "status": "reloaded",
            "source_file": name,
            "previous_version": previous,
            "dataset_version": version,
            "changed": version != previous,
            "rows": int(len(loaded)),
            "seasons": sorted(loaded["season"].astype(str).unique().tolist()),
        }
