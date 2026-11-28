"""Estimate how much a team's offence and its opponent's defence each explain a game.

The baseline blends our offence PPP with the opponent's PPP allowed. This check
asks the same question of real games in the play-by-play shot data: for each
team-game, how well do the team's season offence and the opponent's season
defence predict points per shot attempt? Both season values leave the game
itself out, so the game being predicted never informs its own predictors.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

KEYS = ["SEASON_STR", "GAME_ID", "TEAM_ABBR", "OPP_ABBR"]


def team_game_table(shots: pd.DataFrame, min_shots: int = 40) -> pd.DataFrame:
    """One row per team-game with leave-one-game-out season offence and opponent defence."""
    games = shots.groupby(KEYS, as_index=False).agg(pts=("POINTS", "sum"), n=("POINTS", "size"))
    games = games[games["n"] >= min_shots].copy()
    offense = games.groupby(["SEASON_STR", "TEAM_ABBR"])[["pts", "n"]].sum().add_prefix("off_")
    defense = games.groupby(["SEASON_STR", "OPP_ABBR"])[["pts", "n"]].sum().add_prefix("def_")
    league = games.groupby("SEASON_STR")[["pts", "n"]].sum()
    games = games.join(offense, on=["SEASON_STR", "TEAM_ABBR"]).join(defense, on=["SEASON_STR", "OPP_ABBR"])
    games["league"] = games["SEASON_STR"].map(league["pts"] / league["n"])
    games["offense"] = (games["off_pts"] - games["pts"]) / (games["off_n"] - games["n"])
    games["defense"] = (games["def_pts"] - games["pts"]) / (games["def_n"] - games["n"])
    games["pps"] = games["pts"] / games["n"]
    return games.replace([np.inf, -np.inf], np.nan).dropna(subset=["offense", "defense"])


def _fit(games: pd.DataFrame) -> np.ndarray:
    weights = np.sqrt(games["n"].to_numpy(dtype=float))
    x = np.column_stack([games["offense"] - games["league"], games["defense"] - games["league"]])
    y = (games["pps"] - games["league"]).to_numpy()
    coef, *_ = np.linalg.lstsq(x * weights[:, None], y * weights, rcond=None)
    return coef


def estimate_offense_share(
    shots: pd.DataFrame,
    seasons: Optional[Iterable[str]] = None,
    min_shots: int = 40,
    n_boot: int = 300,
    seed: int = 0,
) -> Dict[str, Any]:
    """Fit points per shot ~ offence edge + defence edge and report the offence share.

    offence share = offence coefficient / (offence + defence coefficients).
    The 95% range comes from resampling team-games with replacement.
    """
    if seasons is not None:
        shots = shots[shots["SEASON_STR"].isin(list(seasons))]
    games = team_game_table(shots, min_shots=min_shots)
    if len(games) < 100:
        raise ValueError("At least 100 team-games are needed to estimate the weights.")
    coef = _fit(games)
    rng = np.random.default_rng(seed)
    shares = []
    for _ in range(n_boot):
        sample = games.iloc[rng.integers(0, len(games), len(games))]
        c = _fit(sample)
        shares.append(c[0] / c.sum())
    y = games["pps"] - games["league"]
    rmse = {}
    for share in (0.5, 0.6, 0.7, 0.8, 1.0):
        pred = share * (games["offense"] - games["league"]) + (1 - share) * (games["defense"] - games["league"])
        rmse[f"{share:.1f}"] = float(np.sqrt(np.average((y - pred) ** 2, weights=games["n"])))
    return {
        "team_games": int(len(games)),
        "seasons": sorted(games["SEASON_STR"].unique().tolist()),
        "min_shots_per_team_game": min_shots,
        "coef_offense": float(coef[0]),
        "coef_defense": float(coef[1]),
        "offense_share": float(coef[0] / coef.sum()),
        "offense_share_95": [float(v) for v in np.percentile(shares, [2.5, 97.5])],
        "weighted_rmse_by_offense_share": rmse,
        "method": "Weighted least squares of team-game points per shot minus the league value on the team's season offence and the opponent's season defence, both relative to league and leaving the game out. Bootstrap over team-games.",
        "scope": "All field-goal attempts, not play-type possessions. Evidence by analogy for the play-type blend.",
    }
