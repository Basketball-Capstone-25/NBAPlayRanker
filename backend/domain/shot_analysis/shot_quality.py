"""Checks required before shot data can support recommendations or models."""
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from domain.shot_analysis.shot_etl import CLEAN_PARQUET, TEAM_ID_TO_ABBR


def inspect_shot_rows(df: pd.DataFrame) -> dict:
    required = {"SEASON_STR", "GAME_ID", "TEAM_ABBR", "OPP_ABBR"}
    missing = sorted(required - set(df.columns))
    if missing:
        return {"valid_attribution": False, "missing_columns": missing, "rows": len(df)}
    teams = set(TEAM_ID_TO_ABBR.values())
    aliases = {"WSH": "WAS", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "NO": "NOP", "GS": "GSW"}
    shooting = df.TEAM_ABBR.replace(aliases)
    opposing = df.OPP_ABBR.replace(aliases)
    self_opponents = shooting.eq(opposing)
    unknown = ~shooting.isin(teams) | ~opposing.isin(teams)
    invalid = unknown | self_opponents
    grouped = df.groupby(["SEASON_STR", "GAME_ID"])
    counts, participants = grouped.size(), grouped.TEAM_ABBR.nunique()
    incomplete = int(participants[counts >= 100].ne(2).sum())
    return {
        "rows": len(df), "invalid_participant_rows": int(invalid.sum()),
        "self_opponent_rows": int(self_opponents.sum()),
        "unsupported_team_rows": int(unknown.sum()),
        "incomplete_game_groups": incomplete,
        "missing_margin_rows": int(df.MARGIN.isna().sum()) if "MARGIN" in df else len(df),
        "location_conflict_rows": int(df.LOCATION_VALID.eq(False).sum()) if "LOCATION_VALID" in df else None,
        "geometry_only_zone_rows": int(df.ZONE_METHOD.eq("geometry_only").sum()) if "ZONE_METHOD" in df else None,
        "valid_attribution": bool(len(df) and not invalid.any() and not incomplete),
    }


@lru_cache(maxsize=2)
def _inspect_file(path: str, size: int, modified: int) -> dict:
    columns = ["SEASON_STR", "GAME_ID", "TEAM_ABBR", "OPP_ABBR", "MARGIN", "LOCATION_VALID", "ZONE_METHOD"]
    present = pq.read_schema(path).names
    return inspect_shot_rows(pd.read_parquet(path, columns=[column for column in columns if column in present]))


def shot_data_quality(path: Path = CLEAN_PARQUET) -> dict:
    try:
        stat = path.stat()
        return _inspect_file(str(path), stat.st_size, stat.st_mtime_ns)
    except (OSError, ValueError, KeyError):
        return {"valid_attribution": False, "reason": "Shot source data is missing or cannot be checked."}
