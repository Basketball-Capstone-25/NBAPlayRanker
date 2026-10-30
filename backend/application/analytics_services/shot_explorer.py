"""SCRUM-627: browse, filter, page and export the play-by-play shot dataset.

Pure rules for the Data Explorer shots view. The shot rows come from the
canonical play-by-play parquet through a loader passed in by the API layer,
so these rules can be tested with a small DataFrame.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
EXPORT_LIMIT = 10_000
RESULTS = {"made": 1, "missed": 0}

# Source column -> (output key, plain-language label), in table order.
COLUMNS: List[Tuple[str, str, str]] = [
    ("GAME_ID", "game_id", "Game"),
    ("PERIOD", "period", "Quarter"),
    ("CLOCK_SEC", "time_left", "Time left"),
    ("TEAM_ABBR", "team", "Team"),
    ("OPP_ABBR", "opponent", "Opponent"),
    ("SHOT_TYPE", "shot_type", "Shot type"),
    ("ZONE", "zone", "Zone"),
    ("IS_MAKE", "result", "Result"),
    ("POINTS", "points", "Points"),
    ("X", "x", "Court x (ft from center)"),
    ("Y", "y", "Court y (ft from center)"),
    ("DIST", "distance_ft", "Distance (ft)"),
]
SORT_KEYS = ["GAME_ID", "PERIOD", "CLOCK_SEC"]


class InvalidShotFilter(ValueError):
    """A filter value the shots view cannot use (HTTP 400)."""


class ShotDataUnavailable(RuntimeError):
    """The shot dataset has not been built or cannot be read (HTTP 503)."""


def _clean(value: Optional[str], name: str, max_len: int = 40) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > max_len:
        raise InvalidShotFilter(f"{name} is too long.")
    return value


@dataclass(frozen=True)
class ShotFilters:
    season: str
    team: str
    opp: Optional[str] = None
    shot_type: Optional[str] = None
    zone: Optional[str] = None
    result: Optional[str] = None
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE

    @classmethod
    def parse(
        cls,
        season: Optional[str],
        team: Optional[str],
        opp: Optional[str] = None,
        shot_type: Optional[str] = None,
        zone: Optional[str] = None,
        result: Optional[str] = None,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> "ShotFilters":
        season = _clean(season, "Season", 7)
        team = _clean(team, "Team", 4)
        if not season or not team:
            raise InvalidShotFilter("Pick a season and team.")
        opp = _clean(opp, "Opponent", 4)
        result = _clean(result, "Result")
        if result is not None:
            result = result.lower()
            if result not in RESULTS:
                raise InvalidShotFilter("Result must be made or missed.")
        if page < 1:
            raise InvalidShotFilter("Page must be 1 or more.")
        if not 1 <= page_size <= MAX_PAGE_SIZE:
            raise InvalidShotFilter(f"Page size must be between 1 and {MAX_PAGE_SIZE}.")
        return cls(
            season=season,
            team=team.upper(),
            opp=opp.upper() if opp else None,
            shot_type=_clean(shot_type, "Shot type"),
            zone=_clean(zone, "Zone"),
            result=result,
            page=page,
            page_size=page_size,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "season": self.season,
            "team": self.team,
            "opp": self.opp,
            "shot_type": self.shot_type,
            "zone": self.zone,
            "result": self.result,
        }


def _cell(key: str, value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if key == "result":
        return "Made" if bool(value) else "Missed"
    if key == "time_left":
        seconds = max(0, int(round(float(value))))
        return f"{seconds // 60}:{seconds % 60:02d}"
    if key in ("period", "points"):
        return int(value)
    if key in ("x", "y", "distance_ft"):
        return round(float(value), 1)
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass(frozen=True)
class ShotExport:
    filters: ShotFilters
    columns: List[Dict[str, str]]
    rows: List[Dict[str, Any]]
    total: int
    capped: bool
    dataset_version: Optional[str]

    def filename(self, ext: str) -> str:
        parts = [self.filters.season, self.filters.team, self.filters.opp or "all"]
        return "pbp_shots_" + "_".join(p.replace(" ", "-") for p in parts) + f".{ext}"

    def to_csv(self) -> str:
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\n")
        filters = "; ".join(f"{k}={v}" for k, v in self.filters.as_dict().items() if v)
        writer.writerow([f"# Play-by-play shots | filters: {filters} | dataset: {self.dataset_version or 'unknown'}"
                         f" | rows: {len(self.rows)} of {self.total}" + (" (capped)" if self.capped else "")])
        writer.writerow([c["label"] for c in self.columns])
        for row in self.rows:
            writer.writerow(["" if row[c["key"]] is None else row[c["key"]] for c in self.columns])
        return out.getvalue()

    def to_json(self) -> Dict[str, Any]:
        return {
            "dataset": "pbp_shots",
            "filters": self.filters.as_dict(),
            "dataset_version": self.dataset_version,
            "total": self.total,
            "exported_rows": len(self.rows),
            "capped": self.capped,
            "export_limit": EXPORT_LIMIT,
            "columns": self.columns,
            "rows": self.rows,
        }


class ShotExplorerService:
    """Applies ShotFilters to the canonical shots and shapes table rows."""

    def __init__(self, load_shots: Callable[[], pd.DataFrame]) -> None:
        self._load_shots = load_shots

    def _shots(self) -> pd.DataFrame:
        try:
            df = self._load_shots()
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise ShotDataUnavailable("Shot data is not available yet.") from exc
        if df is None or not {"SEASON_STR", "TEAM_ABBR"}.issubset(df.columns):
            raise ShotDataUnavailable("Shot data is not available yet.")
        return df

    @staticmethod
    def filter_shots(df: pd.DataFrame, f: ShotFilters) -> pd.DataFrame:
        mask = (df["SEASON_STR"] == f.season) & (df["TEAM_ABBR"] == f.team)
        if f.opp and "OPP_ABBR" in df.columns:
            mask &= df["OPP_ABBR"] == f.opp
        if f.shot_type and "SHOT_TYPE" in df.columns:
            mask &= df["SHOT_TYPE"] == f.shot_type
        if f.zone and "ZONE" in df.columns:
            mask &= df["ZONE"] == f.zone
        if f.result and "IS_MAKE" in df.columns:
            mask &= df["IS_MAKE"].fillna(0).astype(int) == RESULTS[f.result]
        out = df[mask]
        keys = [k for k in SORT_KEYS if k in out.columns]
        if keys:
            ascending = [k != "CLOCK_SEC" for k in keys]
            out = out.sort_values(keys, ascending=ascending, kind="mergesort")
        return out

    @staticmethod
    def columns(df: pd.DataFrame) -> List[Dict[str, str]]:
        return [{"key": key, "label": label} for src, key, label in COLUMNS if src in df.columns]

    @staticmethod
    def to_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
        present = [(src, key) for src, key, _ in COLUMNS if src in df.columns]
        return [{key: _cell(key, rec[src]) for src, key in present} for rec in df.to_dict(orient="records")]

    def page(self, f: ShotFilters) -> Dict[str, Any]:
        df = self._shots()
        matched = self.filter_shots(df, f)
        total = int(len(matched))
        start = (f.page - 1) * f.page_size
        window = matched.iloc[start:start + f.page_size]
        return {
            "filters": f.as_dict(),
            "columns": self.columns(df),
            "rows": self.to_table(window),
            "page": f.page,
            "page_size": f.page_size,
            "total": total,
            "has_next": start + f.page_size < total,
            "dataset_version": df.attrs.get("dataset_version"),
        }

    def export(self, f: ShotFilters) -> ShotExport:
        df = self._shots()
        matched = self.filter_shots(df, f)
        total = int(len(matched))
        return ShotExport(
            filters=f,
            columns=self.columns(df),
            rows=self.to_table(matched.head(EXPORT_LIMIT)),
            total=total,
            capped=total > EXPORT_LIMIT,
            dataset_version=df.attrs.get("dataset_version"),
        )
