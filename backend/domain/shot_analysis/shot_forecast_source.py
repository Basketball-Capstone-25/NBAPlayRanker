"""Prepare dated shot outcomes for forecasts that use only earlier games."""

from datetime import date, datetime
import re

import numpy as np
import pandas as pd

from domain.shot_analysis.shot_etl import TEAM_ID_TO_ABBR

SHOT_COLUMNS = ["EVENT_ID", "GAME_ID", "SEASON_STR", "TEAM_ABBR", "OPP_ABBR", "HOME_FLAG", "POINTS"]
GAME_COLUMNS = ["game_id", "game_date", "season", "season_type"]


def _ids(values: pd.Series, name: str) -> pd.Series:
    result = values.astype("string")
    if not result.str.fullmatch(r"[1-9]\d*").fillna(False).all():
        raise ValueError(f"{name} must contain non-missing numeric identifiers.")
    return result


def _game_date(value) -> pd.Timestamp:
    if isinstance(value, str):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].*)?", value):
            raise ValueError("Game dates must use an ISO calendar date.")
    elif not isinstance(value, (date, datetime, pd.Timestamp)):
        raise ValueError("Game dates are missing or invalid.")
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("Missing game date.")
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert("America/New_York").tz_localize(None)
        return stamp.normalize()
    except (TypeError, ValueError) as error:
        raise ValueError("Game dates are missing or invalid.") from error


def prepare_shot_forecast_rows(shots: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join regular-season outcomes to game dates without retaining current-shot predictors."""
    if not set(SHOT_COLUMNS).issubset(shots) or not set(GAME_COLUMNS).issubset(games):
        raise ValueError("Shot forecast source columns are missing.")
    if shots.empty:
        raise ValueError("Shot forecast source has no rows.")
    clean = shots[SHOT_COLUMNS].copy()
    clean["EVENT_ID"] = _ids(clean.EVENT_ID, "Event IDs")
    clean["GAME_ID"] = _ids(clean.GAME_ID, "Shot game IDs")
    if clean.EVENT_ID.duplicated().any():
        raise ValueError("Duplicate shot events would double-count attempts.")
    teams = set(TEAM_ID_TO_ABBR.values())
    if not clean.TEAM_ABBR.isin(teams).all() or not clean.OPP_ABBR.isin(teams).all() or clean.TEAM_ABBR.eq(clean.OPP_ABBR).any():
        raise ValueError("Shot teams and opponents must be distinct NBA teams.")
    if not clean.HOME_FLAG.isin([False, True]).all():
        raise ValueError("Home flags must be binary and non-missing.")
    clean["HOME_FLAG"] = clean.HOME_FLAG.astype(bool)
    points = pd.to_numeric(clean.POINTS, errors="coerce")
    if not points.isin([0, 2, 3]).all() or clean.POINTS.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError("Shot points must be finite recorded outcomes of 0, 2 or 3.")
    clean["POINTS"] = points.astype("int64")
    participants = pd.DataFrame({"GAME_ID": clean.GAME_ID,
        "home": np.where(clean.HOME_FLAG, clean.TEAM_ABBR, clean.OPP_ABBR),
        "away": np.where(clean.HOME_FLAG, clean.OPP_ABBR, clean.TEAM_ABBR)})
    if participants.groupby("GAME_ID")[["home", "away"]].nunique().gt(1).any().any():
        raise ValueError("A game's participants or home flags disagree.")

    metadata = games[GAME_COLUMNS].drop_duplicates().copy()
    metadata["GAME_ID"] = _ids(metadata.game_id, "Metadata game IDs")
    source_metadata_games = metadata.GAME_ID.nunique()
    # Raw files also contain games absent from the retained shot dataset.
    metadata = metadata.loc[metadata.GAME_ID.isin(clean.GAME_ID)].copy()
    metadata["GAME_DATE"] = metadata.game_date.map(_game_date)
    for column in ("season", "season_type"):
        numbers = pd.to_numeric(metadata[column], errors="coerce")
        if numbers.isna().any() or not np.isfinite(numbers).all() or not numbers.eq(numbers.round()).all():
            raise ValueError("Game season metadata must contain whole numbers.")
        if metadata[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError("Game season metadata must contain whole numbers.")
        metadata[column] = numbers.astype("int64")
    if not metadata.season.between(2001, 2100).all() or not metadata.season_type.isin([1, 2, 3]).all():
        raise ValueError("Game season or season type is unsupported.")
    metadata = metadata[["GAME_ID", "GAME_DATE", "season", "season_type"]].drop_duplicates()
    if metadata.GAME_ID.duplicated().any():
        raise ValueError("Conflicting game metadata cannot establish a forecast date.")
    joined = clean.merge(metadata, on="GAME_ID", how="left", validate="many_to_one")
    if joined.GAME_DATE.isna().any():
        raise ValueError("A shot game has no dated source metadata.")
    seasons = joined.season.astype(int).map(lambda year: f"{year - 1}-{year % 100:02d}")
    if not joined.SEASON_STR.eq(seasons).fillna(False).all():
        raise ValueError("Shot season labels disagree with source game seasons.")
    retained = joined.loc[joined.season_type.eq(2), [*SHOT_COLUMNS, "GAME_DATE"]].reset_index(drop=True)
    coverage = {"source_shots": len(clean), "source_games": int(clean.GAME_ID.nunique()),
        "metadata_games": len(metadata), "regular_season_shots": len(retained),
        "unreferenced_metadata_games": int(source_metadata_games - len(metadata)),
        "regular_season_games": int(retained.GAME_ID.nunique()),
        "excluded_playoff_shots": int(joined.season_type.eq(3).sum()),
        "excluded_other_phase_shots": int(joined.season_type.eq(1).sum())}
    return retained, coverage
