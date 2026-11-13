"""Observed offensive play-type totals across regular seasons."""

import math
import re

from domain.baseline_recommendation import BaselineRecommender
from domain.baseline_recommendation.baseline_recommender import AGGREGATION_VERSION


class TrendDataUnavailable(ValueError):
    pass


def _count(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TrendDataUnavailable("Missing season counts.") from error
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise TrendDataUnavailable("Invalid season counts.")
    return int(number)


def build_season_trend(rec: BaselineRecommender, team: str, play_type: str) -> dict:
    offense = rec.team_df.loc[rec.team_df.SIDE.eq("offense")]
    if team not in set(offense.TEAM_ABBREVIATION):
        raise ValueError("Select a known team.")
    if play_type not in set(offense.PLAY_TYPE):
        raise ValueError("Select a known offensive play type.")
    seasons = set(offense.SEASON)
    if any(not isinstance(season, str) or not re.fullmatch(r"20\d{2}-\d{2}", season)
           or season[-2:] != str(int(season[:4]) + 1)[-2:] for season in seasons):
        raise TrendDataUnavailable("Invalid season labels.")
    seasons = sorted(seasons)
    selected = offense.loc[offense.TEAM_ABBREVIATION.eq(team)]
    if selected.duplicated(["SEASON", "PLAY_TYPE"]).any():
        raise TrendDataUnavailable("Duplicate season totals.")
    rows = []
    # Include absent interior years so the chart cannot join across a data gap.
    for year in range(int(seasons[0][:4]), int(seasons[-1][:4]) + 1):
        season = f"{year}-{(year + 1) % 100:02d}"
        year_rows = selected.loc[selected.SEASON.eq(season)]
        total = sum(_count(value) for value in year_rows.POSS) if len(year_rows) else None
        play = year_rows.loc[year_rows.PLAY_TYPE.eq(play_type)]
        row = {"season": season, "possessions": None, "points": None,
               "team_possessions": total, "usage_pct": None, "ppp": None}
        if len(play):
            possessions, points = _count(play.iloc[0].POSS), _count(play.iloc[0].PTS)
            if not possessions and points:
                raise TrendDataUnavailable("Points require recorded possessions.")
            row.update(possessions=possessions, points=points,
                       usage_pct=100 * possessions / total if total else None,
                       ppp=points / possessions if possessions else None)
        rows.append(row)
    return {
        "status": "ok" if any((r["possessions"] or 0) > 0 for r in rows) else "no_data",
        "filters": {"team": team, "play_type": play_type, "side": "offense", "season_phase": "regular_season"},
        "dataset_version": rec.raw_df.attrs.get("dataset_version", "unavailable"),
        "aggregation_version": AGGREGATION_VERSION,
        "rows": rows,
    }
