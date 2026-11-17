"""Independent possession rates from rounded player-level source values."""

import math
import re

import pandas as pd
from pandas.api.types import is_bool, is_scalar

from domain.baseline_recommendation import BaselineRecommender
from domain.baseline_recommendation.baseline_recommender import AGGREGATION_VERSION

RATE_COLUMNS = {
    "scoring": "SCORE_POSS_PCT", "turnover": "TOV_POSS_PCT",
    "free_throw": "FT_POSS_PCT", "shooting_foul": "SF_POSS_PCT", "and_one": "PLUSONE_POSS_PCT",
}


class OutcomeDataUnavailable(ValueError):
    pass


def _count(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise OutcomeDataUnavailable("Missing possession counts.") from error
    if is_bool(value) or not math.isfinite(number) or not 0 <= number <= 2**53 - 1 or not number.is_integer():
        raise OutcomeDataUnavailable("Invalid possession counts.")
    return int(number)


def _rate(values, weights, total):
    usable = missing = invalid = 0
    weighted = []
    for value, weight in zip(values, weights):
        if weight == 0:
            continue
        if not is_scalar(value) or is_bool(value):
            invalid += weight
            continue
        if pd.isna(value):
            missing += weight
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            invalid += weight
            continue
        if not math.isfinite(number) or not 0 <= number <= 1:
            invalid += weight
            continue
        usable += weight
        weighted.append(number * weight)
    status = "invalid" if invalid else "missing" if missing else "ok" if total else "no_possessions"
    return {"value": math.fsum(weighted) / total if status == "ok" else None, "status": status,
            "usable_possessions": usable, "missing_possessions": missing, "invalid_possessions": invalid}


def build_possession_outcomes(rec: BaselineRecommender, season: str, team: str) -> dict:
    if not re.fullmatch(r"20\d{2}-\d{2}", season) or season[-2:] != str(int(season[:4]) + 1)[-2:]:
        raise ValueError("Use a season such as 2024-25.")
    source = rec.raw_df
    keys = ["SEASON", "PLAYER_ID", "TEAM_ABBREVIATION", "PLAY_TYPE", "TYPE_GROUPING"]
    if not {*keys, "SEASON_ID", "POSS"}.issubset(source.columns):
        raise OutcomeDataUnavailable("Missing possession source columns.")
    phase = source.SEASON_ID.astype(str)
    if not phase.str.fullmatch(r"[24]\d{4}").all():
        raise OutcomeDataUnavailable("Invalid season identifiers.")
    regular = source.loc[phase.str.startswith("2")]
    years = regular.SEASON_ID.astype(str).str[1:].astype(int)
    labels = years.map(lambda year: f"{year}-{(year + 1) % 100:02d}")
    if not regular.SEASON.eq(labels).all() or regular[keys].isna().any().any() or regular.duplicated(keys).any():
        raise OutcomeDataUnavailable("Invalid or duplicate possession source rows.")
    if not regular.TYPE_GROUPING.isin(["Offensive", "Defensive"]).all() or (
        "ENTITY_TYPE" in regular and not regular.ENTITY_TYPE.eq("Player").all()
    ):
        raise OutcomeDataUnavailable("Possession rates require identified player rows.")
    if team not in set(regular.TEAM_ABBREVIATION):
        raise ValueError("Select a known team.")
    if season not in set(regular.SEASON):
        raise ValueError("Select an available season.")
    selected = regular.loc[regular.SEASON.eq(season) & regular.TEAM_ABBREVIATION.eq(team)
                           & regular.TYPE_GROUPING.eq("Offensive")]
    if any(not isinstance(play, str) or not play.strip() or len(play) > 40 for play in selected.PLAY_TYPE):
        raise OutcomeDataUnavailable("Invalid possession categories.")
    if selected.PLAY_TYPE.nunique() > 30:
        raise OutcomeDataUnavailable("Too many possession categories.")
    rows = []
    for play_type, group in selected.groupby("PLAY_TYPE", sort=True):
        weights = [_count(value) for value in group.POSS]
        total = _count(sum(weights))
        rates = {key: _rate(group[column] if column in group else [None] * len(group), weights, total)
                 for key, column in RATE_COLUMNS.items()}
        rows.append({"play_type": play_type, "possessions": total, "rates": rates})
    return {"status": "ok" if rows else "no_data",
            "filters": {"season": season, "team": team, "side": "offense", "season_phase": "regular_season"},
            "dataset_version": source.attrs.get("dataset_version", "unavailable"),
            "aggregation_version": AGGREGATION_VERSION, "rows": rows}
