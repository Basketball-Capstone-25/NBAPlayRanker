"""SCRUM-629: pre-game late-game (clutch) report from the play-by-play shots.

The analyst picks a matchup and season, then narrows the shots to late-game
situations by period, time left in the period and score margin. Margins are
from each shooting team's side, so "trailing by 1 to 5" means the same thing
for both teams in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

MAX_PERIOD = 4  # 4 = fourth quarter; overtime periods are 5 and above
MAX_SECONDS_LEFT = 12 * 60
MAX_MARGIN = 40


class InvalidClutchFilter(ValueError):
    """A filter the clutch report cannot use (HTTP 400)."""


def _text(value: Optional[str], name: str, max_len: int) -> str:
    value = (value or "").strip()
    if not value:
        raise InvalidClutchFilter(f"Pick a {name}.")
    if len(value) > max_len:
        raise InvalidClutchFilter(f"{name.capitalize()} is too long.")
    return value


@dataclass(frozen=True)
class ClutchFilters:
    season: str
    team: str
    opp: str
    period_from: int = 4
    seconds_left: int = 300
    margin_min: int = -5
    margin_max: int = 5

    @classmethod
    def parse(
        cls,
        season: Optional[str],
        team: Optional[str],
        opp: Optional[str],
        period_from: int = 4,
        seconds_left: int = 300,
        margin_min: int = -5,
        margin_max: int = 5,
    ) -> "ClutchFilters":
        season = _text(season, "season", 7)
        team = _text(team, "team", 4).upper()
        opp = _text(opp, "opponent", 4).upper()
        if team == opp:
            raise InvalidClutchFilter("Pick two different teams.")
        if not 1 <= period_from <= MAX_PERIOD:
            raise InvalidClutchFilter(f"Start period must be between 1 and {MAX_PERIOD} (overtime is always included).")
        if not 1 <= seconds_left <= MAX_SECONDS_LEFT:
            raise InvalidClutchFilter("Time left must be between 1 second and 12 minutes.")
        if not (-MAX_MARGIN <= margin_min <= MAX_MARGIN and -MAX_MARGIN <= margin_max <= MAX_MARGIN):
            raise InvalidClutchFilter(f"Score margin must be between -{MAX_MARGIN} and {MAX_MARGIN}.")
        if margin_min > margin_max:
            raise InvalidClutchFilter("The lowest margin must not be above the highest margin.")
        return cls(season, team, opp, period_from, seconds_left, margin_min, margin_max)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "season": self.season,
            "team": self.team,
            "opp": self.opp,
            "period_from": self.period_from,
            "seconds_left": self.seconds_left,
            "margin_min": self.margin_min,
            "margin_max": self.margin_max,
        }


SMALL_SAMPLE = 20  # fewer shots than this in a zone is flagged as a small sample
ZONE_ORDER = ["rim", "paint", "mid", "corner3", "arc3", "unknown"]  # zone codes from shot_config
ZONE_LABELS = {
    "rim": "At the rim",
    "paint": "Paint (not at the rim)",
    "mid": "Mid-range",
    "corner3": "Corner 3",
    "arc3": "Above-the-break 3",
    "unknown": "Unknown location",
}


class ClutchDataUnavailable(RuntimeError):
    """The shot dataset has not been built or is missing columns (HTTP 503)."""


REQUIRED = {"SEASON_STR", "TEAM_ABBR", "OPP_ABBR", "PERIOD", "CLOCK_SEC", "MARGIN", "ZONE", "IS_MAKE"}


def _zone_key(zone: str):
    return (ZONE_ORDER.index(zone), zone) if zone in ZONE_ORDER else (len(ZONE_ORDER), zone)


def _points(df: pd.DataFrame) -> pd.Series:
    made = df["IS_MAKE"].fillna(0).astype(int)
    if "POINTS" in df.columns:
        return df["POINTS"].fillna(0).astype(float) * (made > 0)
    if "SHOT_VALUE" in df.columns:
        return df["SHOT_VALUE"].fillna(0).astype(float) * made
    raise ClutchDataUnavailable("Shot data has no points column.")


def _split(shots: int, makes: int, points: float) -> Dict[str, Any]:
    return {
        "shots": shots,
        "makes": makes,
        "points": int(round(points)),
        "points_per_shot": round(points / shots, 3) if shots else None,
        "small_sample": shots < SMALL_SAMPLE,
    }


def late_game_shots(df: pd.DataFrame, f: ClutchFilters, shooter: str, defender: str) -> pd.DataFrame:
    """One team's shots against the other in the chosen late-game situations.

    MARGIN is already the shooting team's score minus the opponent's before the
    shot, so the same margin range applies to both teams. Shots with an unknown
    margin are left out rather than guessed.
    """
    mask = (
        (df["SEASON_STR"] == f.season)
        & (df["TEAM_ABBR"] == shooter)
        & (df["OPP_ABBR"] == defender)
        & (df["PERIOD"] >= f.period_from)
        & (df["CLOCK_SEC"] <= f.seconds_left)
        & df["MARGIN"].between(f.margin_min, f.margin_max)
    )
    return df[mask]


def build_report(df: pd.DataFrame, f: ClutchFilters) -> Dict[str, Any]:
    if df is None or not REQUIRED.issubset(df.columns):
        raise ClutchDataUnavailable("Shot data is not available yet.")
    sides = {f.team: late_game_shots(df, f, f.team, f.opp), f.opp: late_game_shots(df, f, f.opp, f.team)}
    totals: Dict[str, Dict[str, Any]] = {}
    by_zone: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for team, shots in sides.items():
        points = _points(shots)
        made = shots["IS_MAKE"].fillna(0).astype(int)
        totals[team] = _split(len(shots), int(made.sum()), float(points.sum()))
        for zone, idx in shots.groupby(shots["ZONE"].fillna("Unknown")).groups.items():
            by_zone.setdefault(str(zone), {})[team] = _split(len(idx), int(made.loc[idx].sum()), float(points.loc[idx].sum()))
    empty = _split(0, 0, 0.0)
    rows = [
        {"zone": zone, "label": ZONE_LABELS.get(zone, zone), "team": teams.get(f.team, empty), "opp": teams.get(f.opp, empty)}
        for zone, teams in sorted(by_zone.items(), key=lambda item: _zone_key(item[0]))
    ]
    message = None
    if not rows:
        message = "No shots match these situations. Try an earlier period, more time left or a wider margin."
    return {
        "filters": f.as_dict(),
        "small_sample_threshold": SMALL_SAMPLE,
        "totals": {"team": totals[f.team], "opp": totals[f.opp]},
        "rows": rows,
        "message": message,
        "dataset_version": df.attrs.get("dataset_version"),
    }
