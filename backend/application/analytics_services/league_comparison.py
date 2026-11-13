"""Compare recorded team rates with their same-season league benchmark."""

import math
import re

from domain.baseline_recommendation import BaselineRecommender
from domain.baseline_recommendation.baseline_recommender import AGGREGATION_VERSION

RATES = {"fg_pct": ("FG_PCT", "FGA", 1), "efg_pct": ("EFG_PCT", "FGA", 1.5),
         "tov_pct": ("TOV_POSS_PCT", "POSS", 1), "ft_pct": ("FT_POSS_PCT", "POSS", 1)}


class LeagueDataUnavailable(ValueError):
    pass


def _number(row, column):
    try:
        return float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise LeagueDataUnavailable("Missing or invalid comparison value.") from error


def _sample(row, *, league=False):
    if row is None:
        return None
    counts = {"POSS": _number(row, "LEAGUE_POSS" if league else "POSS"), "FGA": _number(row, "FGA")}
    if any(not math.isfinite(n) or n < 0 or not n.is_integer() for n in counts.values()):
        raise LeagueDataUnavailable("Invalid comparison counts.")
    sample = {"possessions": int(counts["POSS"]), "field_goal_attempts": int(counts["FGA"])}
    for key, (column, denominator, upper) in RATES.items():
        if counts[denominator] == 0:
            sample[key] = None
            continue
        value = _number(row, column)
        if not math.isfinite(value) or not 0 <= value <= upper:
            raise LeagueDataUnavailable("Invalid comparison rates.")
        sample[key] = value
    return sample


def build_league_comparison(rec: BaselineRecommender, season: str, team: str, side: str = "offense") -> dict:
    if not re.fullmatch(r"20\d{2}-\d{2}", season) or season[-2:] != str(int(season[:4]) + 1)[-2:]:
        raise ValueError("Use a season such as 2024-25.")
    if team not in set(rec.team_df["TEAM_ABBREVIATION"]):
        raise ValueError("Select a known team.")
    if side not in {"offense", "defense"}:
        raise ValueError("Choose offense or defense.")
    team_rows = rec.team_df.loc[rec.team_df.SEASON.eq(season) & rec.team_df.SIDE.eq(side)
                                & rec.team_df.TEAM_ABBREVIATION.eq(team)]
    league_rows = rec.league_df.loc[rec.league_df.SEASON.eq(season) & rec.league_df.SIDE.eq(side)]
    if team_rows.PLAY_TYPE.duplicated().any() or league_rows.PLAY_TYPE.duplicated().any():
        raise LeagueDataUnavailable("Duplicate comparison categories.")
    result = {
        "status": "ok" if len(team_rows) else "no_data",
        "filters": {"season": season, "team": team, "side": side, "season_phase": "regular_season"},
        "selected_team_in_league": True,
        "dataset_version": rec.raw_df.attrs.get("dataset_version", "unavailable"),
        "aggregation_version": AGGREGATION_VERSION,
        "rows": [],
    }
    if team_rows.empty:
        return result
    teams = team_rows.set_index("PLAY_TYPE").to_dict("index")
    league = league_rows.set_index("PLAY_TYPE").to_dict("index")
    for play_type in sorted(teams.keys() | league.keys()):
        own, benchmark = _sample(teams.get(play_type)), _sample(league.get(play_type), league=True)
        # Fractions become percentage-point differences; never a percentage change.
        differences = {key: 100 * (own[key] - benchmark[key])
                       if own and benchmark and own[key] is not None and benchmark[key] is not None else None
                       for key in RATES}
        result["rows"].append({"play_type": play_type, "team": own, "league": benchmark,
                               "difference_pp": differences})
    return result
