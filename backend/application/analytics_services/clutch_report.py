"""SCRUM-629: pre-game late-game (clutch) report from the play-by-play shots.

The analyst picks a matchup and season, then narrows the shots to late-game
situations by period, time left in the period and score margin. Margins are
from each shooting team's side, so "trailing by 1 to 5" means the same thing
for both teams in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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
