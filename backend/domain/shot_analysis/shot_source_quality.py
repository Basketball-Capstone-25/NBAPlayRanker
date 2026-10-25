"""Conservative source-event quarantine before deriving shot statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def quarantine_score_discontinuities(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep whole games only when each field goal agrees with both score deltas.

    A mismatch may be a provider correction, duplicate, or out-of-order event.
    Do not choose an event to delete or invent a corrected score. Exclude the
    whole game from the software dataset and retain its IDs for reconciliation.
    """
    columns = {"game_id", "game_play_number", "id", "home_score", "away_score", "team_id", "home_team_id", "score_value", "shooting_play", "type_text"}
    missing = sorted(columns - set(raw.columns))
    if missing:
        raise ValueError(f"Source score audit requires columns: {missing}")
    ordered = raw.sort_values(["game_id", "game_play_number"], kind="stable").copy()
    grouped = ordered.groupby("game_id", sort=False)
    prior_home = grouped.home_score.shift().fillna(0)
    prior_away = grouped.away_score.shift().fillna(0)
    home = ordered.team_id.eq(ordered.home_team_id)
    shots = ordered.shooting_play.astype(bool) & ~ordered.type_text.str.contains("Free Throw", case=False, na=False)
    home_delta = ordered.home_score - prior_home
    away_delta = ordered.away_score - prior_away
    inconsistent = shots & (~np.isclose(home_delta, ordered.score_value.where(home, 0)) | ~np.isclose(away_delta, ordered.score_value.where(~home, 0)))
    bad = ordered.loc[inconsistent]
    bad_games = set(bad.game_id)
    excluded = ordered.game_id.isin(bad_games)
    games = []
    for game_id in sorted(bad_games):
        game_mask = ordered.game_id.eq(game_id)
        events = bad.loc[bad.game_id.eq(game_id)]
        games.append({"game_id": str(game_id), "shot_rows": int((shots & game_mask).sum()), "inconsistent_event_ids": events.id.astype(str).tolist()})
    return raw.loc[~raw.game_id.isin(bad_games)].copy(), {
        "reason": "Field-goal event score does not match the preceding event plus its awarded points; whole game quarantined without guessing which event is correct.",
        "games": games, "excluded_games": len(games),
        "excluded_shots": int((shots & excluded).sum()), "inconsistent_shot_events": int(inconsistent.sum()),
    }
