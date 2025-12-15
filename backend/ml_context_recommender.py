from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline_recommender import BaselineRecommender

# Base folder for all of our data files.
DATA_DIR = Path(__file__).parent / "data"

# Main Synergy snapshot used by both the baseline model and the ML pipeline.
SYNERGY_CSV_PATH = DATA_DIR / "synergy_playtypes_2019_2025_players.csv"

# Offline ML predictions (RandomForest PPP) produced in a separate notebook/script.
ML_PRED_PATH = DATA_DIR / "ml_offense_ppp_predictions.csv"


def build_ml_matchup_table(
    season: str,
    our_team: str,
    opp_team: str,
    w_off: float = 0.7,
    w_def: float = 0.3,
) -> pd.DataFrame:
    """
    Build a matchup table that uses ML PPP predictions instead of baseline PPP.

    What it returns:
      - One row per play type for (season, our_team vs opp_team).
      - For each row we keep:
          * PPP_ML: RandomForest PPP prediction for our offense.
          * PPP_DEF_SHRUNK: opponent defensive PPP shrunk toward league.
          * PPP_PRED_ML: combined offense/defense matchup score.
          * PPP_GAP_ML: PPP_ML minus PPP_DEF_SHRUNK.

    How it works:
      1) Reuse the pre-processing from the BaselineRecommender to get team_df and
         league_df (team and league tables).
      2) Merge in the offline ML predictions (PPP_ML).
      3) Slice out the offensive rows for our team and defensive rows for the opponent.
      4) Compute PPP_DEF_SHRUNK using league defense as an anchor.
      5) Build a combined matchup score PPP_PRED_ML.
    """
    # Use the same pipeline as the baseline model to build team/league tables.
    rec = BaselineRecommender(str(SYNERGY_CSV_PATH))
    team_df = rec.team_df.copy()
    league_df = rec.league_df.copy()

    # Load precomputed ML PPP predictions for offense.
    if not ML_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing ML prediction file: {ML_PRED_PATH}")
    ml_df = pd.read_csv(ML_PRED_PATH)

    # Attach PPP_ML to the team-level table using season + team + play type.
    team_df = team_df.merge(
        ml_df,
        on=["SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE"],
        how="left",
    )

    # Offense rows for our team in this season.
    off = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @our_team and SIDE == 'offense'"
    ).copy()

    # Defense rows for the opponent in this season.
    deff = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @opp_team and SIDE == 'defense'"
    ).copy()

    if off.empty:
        raise ValueError(f"No offensive data for team {our_team} in season {season}")
    if deff.empty:
        raise ValueError(f"No defensive data for team {opp_team} in season {season}")

    # League offense baselines by play type for this season.
    league_off = league_df.query(
        "SEASON == @season and SIDE == 'offense'"
    )[["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]].rename(
        columns={"PPP": "PPP_LEAGUE_OFF", "RELIABILITY_WEIGHT": "REL_OFF_LEAGUE"}
    )

    # League defense baselines by play type for this season.
    league_def = league_df.query(
        "SEASON == @season and SIDE == 'defense'"
    )[["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]].rename(
        columns={"PPP": "PPP_LEAGUE_DEF", "RELIABILITY_WEIGHT": "REL_DEF_LEAGUE"}
    )

    # Subset of opponent defense columns we need for the matchup.
    deff_subset = deff[
        [
            "PLAY_TYPE",
            "PPP",
            "POSS",
            "POSS_PCT",
            "RELIABILITY_WEIGHT",
            "EFG_PCT",
            "SCORE_POSS_PCT",
            "TOV_POSS_PCT",
        ]
    ].copy()

    # Join our offense with their defense on play type.
    merged = off.merge(
        deff_subset,
        on="PLAY_TYPE",
        suffixes=("_OFF", "_DEF"),
    )

    # Add league offense/defense baselines into the same table.
    merged = merged.merge(league_off, on="PLAY_TYPE", how="left")
    merged = merged.merge(league_def, on="PLAY_TYPE", how="left")

    # Shrink opponent defense PPP toward league defense PPP.
    # Same shrinkage idea as baseline: more possessions => we trust team more.
    rel_def = merged["RELIABILITY_WEIGHT_DEF"]
    merged["PPP_DEF_SHRUNK"] = (
        rel_def * merged["PPP_DEF"] + (1 - rel_def) * merged["PPP_LEAGUE_DEF"]
    )

    # PPP_ML should be present after the earlier merge; if not, something is off.
    if "PPP_ML" not in merged.columns:
        raise ValueError("PPP_ML column missing after merge. Check ML predictions file.")

    # Drop any play types that did not receive an ML prediction.
    merged = merged[merged["PPP_ML"].notna()].copy()

    # Combined ML-based matchup score (offense vs defense).
    #
    # Same shape as the baseline formula:
    #   - w_off scales how much we lean on the offensive prediction.
    #   - w_def adjusts for how friendly/unfriendly the opponent defense is
    #     compared to league.
    merged["PPP_PRED_ML"] = (
        w_off * merged["PPP_ML"]
        + w_def * (2 * merged["PPP_LEAGUE_OFF"] - merged["PPP_DEF_SHRUNK"])
    )

    # Simple gap metric: how much better we expect to do vs what they usually allow.
    merged["PPP_GAP_ML"] = merged["PPP_ML"] - merged["PPP_DEF_SHRUNK"]

    merged = merged.sort_values("PPP_PRED_ML", ascending=False).reset_index(drop=True)
    return merged


# Play-type priority weights for context logic.
#
# These are small hand-crafted weights that say, “If I need a 3, which plays
# are naturally better at generating threes?” or “If I need something fast,
# which plays are quick hitters?”.
#
# Higher weight = higher priority within that group.
THREE_PT_WEIGHTS = {
    "Spotup": 1.0,
    "OffScreen": 0.8,
    "Isolation": 0.6,
    "PRBallHandler": 0.4,
}

QUICK_WEIGHTS = {
    "Spotup": 1.0,
    "OffScreen": 0.8,
    "Cut": 0.6,
    "Isolation": 0.4,
}


def add_playtype_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add play-type flags used by the context logic.

    What it adds:
      - THREE_PT_PRIORITY / QUICK_PRIORITY: numeric weights (0 if not in group).
      - IS_3PT_ORIENTED / IS_QUICK: 0/1 flags based on those weights.
      - IS_SLOW_2: 1 if the play is neither 3PT-oriented nor quick.
    """
    df = df.copy()

    # Map play types into three-point and quick weights; unknown play types default to 0.
    df["THREE_PT_PRIORITY"] = df["PLAY_TYPE"].map(THREE_PT_WEIGHTS).fillna(0.0)
    df["QUICK_PRIORITY"] = df["PLAY_TYPE"].map(QUICK_WEIGHTS).fillna(0.0)

    # Simple binary flags to make it easy to ask “is this a 3pt-type play?” etc.
    df["IS_3PT_ORIENTED"] = (df["THREE_PT_PRIORITY"] > 0).astype(int)
    df["IS_QUICK"] = (df["QUICK_PRIORITY"] > 0).astype(int)

    # "Slow 2" = not 3PT-oriented and not quick.
    # These are the plays we might want to penalize when urgency is high.
    df["IS_SLOW_2"] = (
        (df["THREE_PT_PRIORITY"] == 0.0) & (df["QUICK_PRIORITY"] == 0.0)
    ).astype(int)

    return df


def total_time_remaining(period: int, time_remaining_period_sec: float) -> float:
    """
    Return total seconds left in a 4x12 minute game.

    period: 1–4
    time_remaining_period_sec: seconds left in the current period (0–720)

    How it works:
      - Convert (period, time left in that period) into a single “seconds
        remaining in the whole game” number.
    """
    total_game = 4 * 12 * 60  # 2880 seconds in regulation
    elapsed = (period - 1) * 12 * 60 + (12 * 60 - time_remaining_period_sec)
    return max(0.0, total_game - elapsed)


def compute_urgencies(margin: float, T_left: float) -> tuple[float, float]:
    """
    Compute urgency levels for 3PT-oriented plays and quick plays.

    margin: our_score - opp_score (positive if we are leading).
    T_left: total seconds remaining in the game.

    Returns:
      (three_urgency, quick_urgency), each in [0, 1].

    Intuition:
      - Earlier in the game (big T_left), urgencies should be low.
      - Late in the game, especially when losing, urgencies should go up.
    """
    # Time pressure: 0 when the game just started, 1 when we are at the end.
    time_pressure = 1.0 - (T_left / 2880.0)
    time_pressure = min(max(time_pressure, 0.0), 1.0)

    # Clip margin so extreme blowouts don't explode the math.
    m = max(-20.0, min(20.0, margin))

    # Margin pressure: more pressure when tied or losing, less when up.
    if m <= 0:
        # Tied or losing: worse margin => more pressure (up to 1 at -20).
        margin_pressure = -m / 20.0
    else:
        # Slight lead still has some pressure; by +5 or more it goes to 0.
        margin_pressure = max(0.0, (5.0 - m) / 5.0)

    # Three-urgency: mostly driven by scoreboard pressure,
    # with a little extra from time pressure.
    three_urgency = 0.8 * margin_pressure + 0.2 * time_pressure
    three_urgency = min(max(three_urgency, 0.0), 1.0)

    # Quick-urgency: mostly driven by time pressure.
    # If we are up by a lot, we don’t need to rush as much.
    quick_urgency = time_pressure * (1.0 - max(0.0, m) / 20.0)
    quick_urgency = min(max(quick_urgency, 0.0), 1.0)

    return three_urgency, quick_urgency


def rank_ml_with_context(
    season: str,
    our_team: str,
    opp_team: str,
    margin: float,
    period: int,
    time_remaining_period_sec: float,
    k: int = 5,
    w_off: float = 0.7,
    w_def: float = 0.3,
    alpha_three: float = 0.05,
    alpha_quick: float = 0.05,
    alpha_penalty: float = 0.08,
) -> pd.DataFrame:
    """
    Rank play types using ML PPP plus game context.

    Inputs:
      - season, our_team, opp_team: which matchup.
      - margin: our_score - opp_score (positive if we are leading).
      - period: 1–4 (we treat OT as 5 in the API layer).
      - time_remaining_period_sec: seconds left in current period (0–720).

    What it does:
      - Starts from the ML matchup table (PPP_PRED_ML).
      - Adds flags for 3-pt, quick, and “slow 2” plays.
      - Computes urgency scores from margin + time.
      - Applies small context bonuses/penalties on top of PPP_PRED_ML.
      - Returns the top-k play types by CONTEXT_SCORE.
    """
    # Base ML matchup scores with no context applied yet.
    df = build_ml_matchup_table(
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        w_off=w_off,
        w_def=w_def,
    )

    # Add play-type flags (3PT, quick, slow-2).
    df = add_playtype_flags(df)

    # Compute total time left in the game and urgency levels.
    T_left = total_time_remaining(period, time_remaining_period_sec)
    three_urgency, quick_urgency = compute_urgencies(margin, T_left)

    # Base score from the ML matchup model.
    base = df["PPP_PRED_ML"].to_numpy(dtype=float)

    # Context bonus:
    #   - alpha_three * three_urgency * THREE_PT_PRIORITY
    #   - alpha_quick * quick_urgency * QUICK_PRIORITY
    #
    # When urgencies are 0, this does nothing; when urgencies are 1 and the
    # play has high priority, it bumps the score a bit.
    context_boost = (
        alpha_three * three_urgency * df["THREE_PT_PRIORITY"].to_numpy(dtype=float)
        + alpha_quick * quick_urgency * df["QUICK_PRIORITY"].to_numpy(dtype=float)
    )

    # Penalty for "slow 2" plays when urgency is high.
    slow_mask = df["IS_SLOW_2"].to_numpy(dtype=float)

    # Combine both urgencies for the penalty term.
    # We weight quick_urgency a bit more since late-game clock is critical.
    combined_urgency = 0.7 * three_urgency + 1.0 * quick_urgency
    penalty = alpha_penalty * combined_urgency * slow_mask

    # Final context-aware score.
    df["CONTEXT_SCORE"] = base + context_boost - penalty

    # Human-readable explanation per play type.
    def explain(row: pd.Series) -> str:
        delta = row["CONTEXT_SCORE"] - row["PPP_PRED_ML"]
        return (
            f"{row['PLAY_TYPE']}: base {row['PPP_PRED_ML']:.3f} PPP, "
            f"{delta:+.3f} from context "
            f"(3-urg={three_urgency:.2f}, quick-urg={quick_urgency:.2f})"
        )

    df["CONTEXT_RATIONALE"] = df.apply(explain, axis=1)

    df = df.sort_values("CONTEXT_SCORE", ascending=False).reset_index(drop=True)
    return df.head(k)
