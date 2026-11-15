"""Pre-game shot features from completed games on strictly earlier dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "PRIOR_TEAM_PPS", "PRIOR_OPPONENT_ALLOWED_PPS", "PRIOR_LEAGUE_PPS", "HOME_FLAG",
]
KEY_COLUMNS = ["GAME_ID", "TEAM_ABBR", "SEASON_STR", "GAME_DATE", "OPP_ABBR", "HOME_FLAG"]
PROOF_COLUMNS = [
    "HISTORY_CUTOFF_DATE", "PRIOR_TEAM_FGA", "PRIOR_TEAM_POINTS", "PRIOR_TEAM_LAST_DATE",
    "PRIOR_OPPONENT_ALLOWED_FGA", "PRIOR_OPPONENT_ALLOWED_POINTS", "PRIOR_OPPONENT_ALLOWED_LAST_DATE",
    "PRIOR_LEAGUE_FGA", "PRIOR_LEAGUE_POINTS", "TEAM_HISTORY_FALLBACK", "OPPONENT_HISTORY_FALLBACK",
]
TARGET_COLUMNS = ["FGA", "POINTS", "ZERO_POINT_SHOTS", "TWO_POINT_SHOTS", "THREE_POINT_SHOTS", "TARGET_PPS"]


def _game_rows(shots: pd.DataFrame) -> pd.DataFrame:
    required = ["EVENT_ID", *KEY_COLUMNS, "POINTS"]
    if not set(required).issubset(shots.columns) or shots.empty:
        raise ValueError("Prepared shots require event, game, team, date and point fields.")
    frame = shots[required].copy()
    for name in ["EVENT_ID", "GAME_ID", "TEAM_ABBR", "OPP_ABBR", "SEASON_STR"]:
        if not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{name} must contain nonempty text.")
    if frame.EVENT_ID.duplicated().any():
        raise ValueError("Duplicate shot event IDs would double-count attempts.")
    if frame.TEAM_ABBR.eq(frame.OPP_ABBR).any():
        raise ValueError("A shooting team cannot be its own opponent.")
    if not frame.SEASON_STR.str.fullmatch(r"\d{4}-\d{2}").all():
        raise ValueError("Season labels must use YYYY-YY.")
    try:
        dates = pd.to_datetime(frame.GAME_DATE, format="ISO8601", errors="raise")
        naive_dates = dates.dt.tz is None
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("GAME_DATE must be an Eastern calendar date without a timezone.") from exc
    if not naive_dates or dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("GAME_DATE must be an Eastern calendar date without a timezone.")
    frame["GAME_DATE"] = dates
    if not frame.HOME_FLAG.isin([True, False]).all():
        raise ValueError("HOME_FLAG must be boolean or zero/one.")
    if frame.POINTS.map(lambda value: isinstance(value, (bool, np.bool_))).any() or not frame.POINTS.isin([0, 2, 3]).all():
        raise ValueError("Shot points must be 0, 2 or 3.")
    frame["POINTS"] = frame.POINTS.astype("int64")
    frame["HOME_FLAG"] = frame.HOME_FLAG.astype(bool)
    by_game = frame.groupby("GAME_ID")[["GAME_DATE", "SEASON_STR"]].nunique()
    by_team = frame.groupby(["GAME_ID", "TEAM_ABBR"])[["OPP_ABBR", "HOME_FLAG"]].nunique()
    if by_game.gt(1).any().any() or by_team.gt(1).any().any():
        raise ValueError("Each game and team must have consistent date, season, opponent and home status.")
    for points, column in [(0, "ZERO_POINT_SHOTS"), (2, "TWO_POINT_SHOTS"), (3, "THREE_POINT_SHOTS")]:
        frame[column] = frame.POINTS.eq(points).astype("int64")
    games = frame.groupby(KEY_COLUMNS, as_index=False).agg(
        FGA=("EVENT_ID", "size"), POINTS=("POINTS", "sum"),
        ZERO_POINT_SHOTS=("ZERO_POINT_SHOTS", "sum"), TWO_POINT_SHOTS=("TWO_POINT_SHOTS", "sum"),
        THREE_POINT_SHOTS=("THREE_POINT_SHOTS", "sum"),
    )
    games["TARGET_PPS"] = games.POINTS / games.FGA
    return games.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBR"]).reset_index(drop=True)


def build_pregame_rows(shots: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Predict each date before adding any of that date's outcomes to history.

    Input is already restricted to regular-season shots. Historical points per
    attempt are weighted by attempts across seasons, never averaged by game.
    """
    games = _game_rows(shots)
    offense, allowed = {}, {}
    league_fga, league_points, cutoff = 0, 0, None
    rows, warmup_dates = [], []
    for date, day in games.groupby("GAME_DATE", sort=True):
        if league_fga == 0:
            warmup_dates.append(date.date().isoformat())
        else:
            league_pps = league_points / league_fga
            for game in day.itertuples(index=False):
                team_fga, team_points, team_date = offense.get(game.TEAM_ABBR, (0, 0, pd.NaT))
                opp_fga, opp_points, opp_date = allowed.get(game.OPP_ABBR, (0, 0, pd.NaT))
                rows.append({
                    **game._asdict(),
                    "PRIOR_TEAM_PPS": team_points / team_fga if team_fga else league_pps,
                    "PRIOR_OPPONENT_ALLOWED_PPS": opp_points / opp_fga if opp_fga else league_pps,
                    "PRIOR_LEAGUE_PPS": league_pps,
                    "HISTORY_CUTOFF_DATE": cutoff,
                    "PRIOR_TEAM_FGA": team_fga, "PRIOR_TEAM_POINTS": team_points, "PRIOR_TEAM_LAST_DATE": team_date,
                    "PRIOR_OPPONENT_ALLOWED_FGA": opp_fga, "PRIOR_OPPONENT_ALLOWED_POINTS": opp_points,
                    "PRIOR_OPPONENT_ALLOWED_LAST_DATE": opp_date,
                    "PRIOR_LEAGUE_FGA": league_fga, "PRIOR_LEAGUE_POINTS": league_points,
                    "TEAM_HISTORY_FALLBACK": team_fga == 0, "OPPONENT_HISTORY_FALLBACK": opp_fga == 0,
                })
        # Updating only after every prediction excludes the whole current date.
        for game in day.itertuples(index=False):
            for history, team in [(offense, game.TEAM_ABBR), (allowed, game.OPP_ABBR)]:
                prior_fga, prior_points, _ = history.get(team, (0, 0, None))
                history[team] = (prior_fga + game.FGA, prior_points + game.POINTS, date)
            league_fga += game.FGA
            league_points += game.POINTS
        cutoff = date
    result = pd.DataFrame(rows, columns=[*KEY_COLUMNS, *TARGET_COLUMNS, *FEATURE_COLUMNS[:-1], *PROOF_COLUMNS])
    by_season = []
    for season, group in games.groupby("SEASON_STR", sort=True):
        scored = result.loc[result.SEASON_STR.eq(season)]
        by_season.append({
            "season": season, "input_game_team_rows": len(group), "input_shots": int(group.FGA.sum()),
            "scored_game_team_rows": len(scored), "scored_shots": int(scored.FGA.sum()),
            "warmup_game_team_rows": len(group) - len(scored), "warmup_shots": int(group.FGA.sum() - scored.FGA.sum()),
            "team_fallback_rows": int(scored.TEAM_HISTORY_FALLBACK.sum()),
            "opponent_fallback_rows": int(scored.OPPONENT_HISTORY_FALLBACK.sum()),
        })
    coverage = {
        "input_games": int(games.GAME_ID.nunique()), "warmup_dates": warmup_dates, "by_season": by_season,
        **{name: sum(season[name] for season in by_season) for name in by_season[0] if name != "season"},
    }
    return result, coverage
