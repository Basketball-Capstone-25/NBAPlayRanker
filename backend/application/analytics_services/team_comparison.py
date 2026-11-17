"""Compare two offensive season profiles on their own full-team totals."""

import math
import re

from pandas.api.types import is_bool

from domain.baseline_recommendation import BaselineRecommender
from domain.baseline_recommendation.baseline_recommender import AGGREGATION_VERSION


class TeamComparisonDataUnavailable(ValueError):
    pass


def _count(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TeamComparisonDataUnavailable("Missing profile counts.") from error
    if is_bool(value) or not math.isfinite(number) or not 0 <= number <= 2**53 - 1 or not number.is_integer():
        raise TeamComparisonDataUnavailable("Invalid profile counts.")
    return int(number)


def _profile(rows):
    if rows.empty:
        return None, {}
    if rows.PLAY_TYPE.duplicated().any():
        raise TeamComparisonDataUnavailable("Duplicate profile categories.")
    samples = {}
    for row in rows.itertuples():
        if not isinstance(row.PLAY_TYPE, str) or not row.PLAY_TYPE.strip() or len(row.PLAY_TYPE) > 40:
            raise TeamComparisonDataUnavailable("Invalid profile categories.")
        possessions, points = _count(row.POSS), _count(row.PTS)
        if possessions == 0 and points > 0:
            raise TeamComparisonDataUnavailable("Points require recorded possessions.")
        samples[row.PLAY_TYPE] = {"possessions": possessions, "points": points,
                                  "ppp": points / possessions if possessions else None}
    total = _count(sum(sample["possessions"] for sample in samples.values()))
    for sample in samples.values():
        sample["usage_pct"] = 100 * sample["possessions"] / total if total else None
    return total, samples


def build_team_comparison(rec: BaselineRecommender, season: str, team: str, opponent: str) -> dict:
    if not re.fullmatch(r"20\d{2}-\d{2}", season) or season[-2:] != str(int(season[:4]) + 1)[-2:]:
        raise ValueError("Use a season such as 2024-25.")
    if team == opponent:
        raise ValueError("Choose two different teams.")
    source = rec.team_df
    columns = {"SEASON", "TEAM_ABBREVIATION", "SIDE", "PLAY_TYPE", "POSS", "PTS"}
    if not columns.issubset(source.columns):
        raise TeamComparisonDataUnavailable("Missing profile columns.")
    if team not in set(source.TEAM_ABBREVIATION) or opponent not in set(source.TEAM_ABBREVIATION):
        raise ValueError("Select known teams.")
    if season not in set(source.SEASON):
        raise ValueError("Select an available season.")
    offense = source.loc[source.SEASON.eq(season) & source.SIDE.eq("offense")]
    team_total, own = _profile(offense.loc[offense.TEAM_ABBREVIATION.eq(team)])
    opponent_total, other = _profile(offense.loc[offense.TEAM_ABBREVIATION.eq(opponent)])
    categories = sorted(own.keys() | other.keys())
    if len(categories) > 30:
        raise TeamComparisonDataUnavailable("Too many profile categories.")
    rows = []
    for play_type in categories:
        left, right = own.get(play_type), other.get(play_type)
        difference = {name: left[field] - right[field]
                      if left and right and left[field] is not None and right[field] is not None else None
                      for name, field in [("usage_pp", "usage_pct"), ("ppp", "ppp")]}
        rows.append({"play_type": play_type, "team": left, "opponent": right, "difference": difference})
    return {
        "status": "ok" if own and other else "no_data",
        "filters": {"season": season, "team": team, "opponent": opponent,
                    "side": "offense", "season_phase": "regular_season"},
        "dataset_version": rec.raw_df.attrs.get("dataset_version", "unavailable"),
        "aggregation_version": AGGREGATION_VERSION,
        "team_possessions": team_total,
        "opponent_possessions": opponent_total,
        "rows": rows,
    }
