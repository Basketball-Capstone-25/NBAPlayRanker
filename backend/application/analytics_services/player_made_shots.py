"""Where a player made shots against one opponent (SCRUM-825).

Rows are the cleaned shots (one per attempt). Only made attempts by the chosen
player, for our team, against the opponent, in the chosen seasons are kept,
then placed on the attacking half court the same way as the shot heatmap.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from domain.shot_analysis.shooter_names import normalize_athlete_id

REQUIRED = ["SEASON_STR", "TEAM_ABBR", "OPP_ABBR", "SHOOTER_ID", "MADE", "GAME_ID", "X", "Y"]


class PlayerShotsUnavailable(Exception):
    """The shot data or the player names are missing."""


class InvalidPlayerShotsFilter(ValueError):
    """A filter the data cannot answer."""


def _check(shots: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in shots.columns]
    if missing:
        raise PlayerShotsUnavailable(f"Shot data is missing columns: {', '.join(missing)}.")


def parse_seasons(raw: str, available: Sequence[str]) -> List[str]:
    """Deduplicate the comma-separated seasons and reject any missing from the available shot data."""
    seasons = sorted({s.strip() for s in raw.split(",") if s.strip()})
    if not seasons:
        raise InvalidPlayerShotsFilter("Choose at least one season.")
    unknown = [s for s in seasons if s not in set(available)]
    if unknown:
        raise InvalidPlayerShotsFilter(f"No shot data for season {', '.join(unknown)}.")
    return seasons


def _matchup(shots: pd.DataFrame, our: str, opp: str, seasons: Sequence[str]) -> pd.DataFrame:
    if our == opp:
        raise InvalidPlayerShotsFilter("Our team and opponent must be different.")
    rows = shots.loc[shots["SEASON_STR"].isin(seasons) & shots["TEAM_ABBR"].eq(our) & shots["OPP_ABBR"].eq(opp)]
    ids = rows["SHOOTER_ID"].map(normalize_athlete_id)
    return rows.assign(_PLAYER=ids)


def _made(rows: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(rows["MADE"], errors="coerce").fillna(0).astype(int).eq(1)


def list_players(shots: pd.DataFrame, names: Dict[str, str], our: str, opp: str, seasons: Sequence[str]) -> dict:
    """Our players who took shots in these games, most makes first, by name."""
    _check(shots)
    rows = _matchup(shots, our, opp, seasons)
    rows = rows.loc[rows["_PLAYER"].notna()]
    made = rows.loc[_made(rows)].groupby("_PLAYER").size()
    players, unnamed = [], 0
    for pid in sorted(rows["_PLAYER"].unique()):
        name = names.get(pid)
        if not name:
            unnamed += 1
            continue
        players.append({"id": pid, "name": name, "made": int(made.get(pid, 0))})
    players.sort(key=lambda p: (-p["made"], p["name"]))
    return {"our": our, "opp": opp, "seasons": list(seasons), "players": players, "unnamed_players": unnamed}


def player_made_shots(shots: pd.DataFrame, names: Dict[str, str], our: str, opp: str, seasons: Sequence[str], player: str) -> dict:
    """Court positions of the player's made shots against the opponent."""
    # Imported here: the court module loads plotting libraries the app avoids at startup.
    from infrastructure.visualization_and_export.viz_shot_heatmap import heatmap_plot_data

    _check(shots)
    pid = normalize_athlete_id(player)
    if not pid or pid not in names:
        raise InvalidPlayerShotsFilter("Choose a player from the list.")
    rows = _matchup(shots, our, opp, seasons)
    games = int(rows["GAME_ID"].nunique())
    mine = rows.loc[rows["_PLAYER"].eq(pid)]
    made = mine.loc[_made(mine)]
    placed, x, y, counts = heatmap_plot_data(made)
    return {
        "player": {"id": pid, "name": names[pid]},
        "our": our,
        "opp": opp,
        "seasons": list(seasons),
        "games": games,
        "player_games": int(mine["GAME_ID"].nunique()),
        "attempts": int(len(mine)),
        "made": int(len(made)),
        "shown": int(len(placed)),
        "excluded_location": counts["n_shots_excluded_location"],
        "outside_halfcourt": counts["n_shots_outside_halfcourt"],
        # Feet: x from midcourt (0) to the baseline (47), basket at 41.75; y from -25 to 25.
        "points": [[round(float(a), 1), round(float(b), 1)] for a, b in zip(x, y)],
    }
