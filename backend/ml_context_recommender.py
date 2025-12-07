from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline_recommender import BaselineRecommender

# File paths for the Synergy data and offline ML predictions
DATA_DIR = Path(__file__).parent / "data"
SYNERGY_CSV_PATH = DATA_DIR / "synergy_playtypes_2019_2025_players.csv"
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

    Each row is a play type for (season, our_team vs opp_team) with:
      - PPP_ML: RandomForest PPP prediction for our offense
      - PPP_DEF_SHRUNK: opponent defense PPP (shrunk toward league)
      - PPP_PRED_ML: combined offense/defense matchup score
      - PPP_GAP_ML: difference between our PPP_ML and their PPP_DEF_SHRUNK
    """
    # Reuse the pre-processing from the baseline recommender
    rec = BaselineRecommender(str(SYNERGY_CSV_PATH))
    team_df = rec.team_df.copy()
    league_df = rec.league_df.copy()

    # Load precomputed ML PPP predictions (offense)
    if not ML_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing ML prediction file: {ML_PRED_PATH}")
    ml_df = pd.read_csv(ML_PRED_PATH)

    # Add PPP_ML to the team-level table
    team_df = team_df.merge(
        ml_df,
        on=["SEASON", "TEAM_ABBREVIATION", "PLAY_TYPE"],
        how="left",
    )

    # Slice offense/defense rows for this matchup
    off = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @our_team and SIDE == 'offense'"
    ).copy()
    deff = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @opp_team and SIDE == 'defense'"
    ).copy()

    if off.empty:
        raise ValueError(f"No offensive data for team {our_team} in season {season}")
    if deff.empty:
        raise ValueError(f"No defensive data for team {opp_team} in season {season}")

    # League PPP by play type for this season (offense and defense)
    league_off = league_df.query(
        "SEASON == @season and SIDE == 'offense'"
    )[["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]].rename(
        columns={"PPP": "PPP_LEAGUE_OFF", "RELIABILITY_WEIGHT": "REL_OFF_LEAGUE"}
    )

    league_def = league_df.query(
        "SEASON == @season and SIDE == 'defense'"
    )[["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]].rename(
        columns={"PPP": "PPP_LEAGUE_DEF", "RELIABILITY_WEIGHT": "REL_DEF_LEAGUE"}
    )

    # Opponent defense columns used in the matchup score
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

    # Join our offense with their defense for each play type
    merged = off.merge(
        deff_subset,
        on="PLAY_TYPE",
        suffixes=("_OFF", "_DEF"),
    )

    # Add league offense/defense baselines
    merged = merged.merge(league_off, on="PLAY_TYPE", how="left")
    merged = merged.merge(league_def, on="PLAY_TYPE", how="left")

    # Shrink opponent defense PPP toward league PPP
    rel_def = merged["RELIABILITY_WEIGHT_DEF"]
    merged["PPP_DEF_SHRUNK"] = (
        rel_def * merged["PPP_DEF"] + (1 - rel_def) * merged["PPP_LEAGUE_DEF"]
    )

    # PPP_ML comes from the offline ML pipeline
    if "PPP_ML" not in merged.columns:
        raise ValueError("PPP_ML column missing after merge. Check ML predictions file.")

    # Drop play types that did not receive an ML prediction
    merged = merged[merged["PPP_ML"].notna()].copy()

    # Combined ML-based matchup score (offense vs defense)
    merged["PPP_PRED_ML"] = (
        w_off * merged["PPP_ML"]
        + w_def * (2 * merged["PPP_LEAGUE_OFF"] - merged["PPP_DEF_SHRUNK"])
    )

    # Simple gap metric: our ML PPP minus their shrunk allowed PPP
    merged["PPP_GAP_ML"] = merged["PPP_ML"] - merged["PPP_DEF_SHRUNK"]

    merged = merged.sort_values("PPP_PRED_ML", ascending=False).reset_index(drop=True)
    return merged


# Play-type priority weights for context logic
# Higher weight = higher priority within that group
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

    Adds:
      - THREE_PT_PRIORITY / QUICK_PRIORITY (0 if not in that group)
      - IS_3PT_ORIENTED / IS_QUICK (binary flags)
      - IS_SLOW_2 (neither 3PT-oriented nor quick)
    """
    df = df.copy()

    df["THREE_PT_PRIORITY"] = df["PLAY_TYPE"].map(THREE_PT_WEIGHTS).fillna(0.0)
    df["QUICK_PRIORITY"] = df["PLAY_TYPE"].map(QUICK_WEIGHTS).fillna(0.0)

    df["IS_3PT_ORIENTED"] = (df["THREE_PT_PRIORITY"] > 0).astype(int)
    df["IS_QUICK"] = (df["QUICK_PRIORITY"] > 0).astype(int)

    # "Slow 2" = not 3PT-oriented and not quick
    df["IS_SLOW_2"] = (
        (df["THREE_PT_PRIORITY"] == 0.0) & (df["QUICK_PRIORITY"] == 0.0)
    ).astype(int)

    return df


def total_time_remaining(period: int, time_remaining_period_sec: float) -> float:
    """
    Return total seconds left in a 4x12 minute game.

    period: 1–4
    time_remaining_period_sec: seconds left in the current period (0–720)
    """
    total_game = 4 * 12 * 60  # 2880 seconds in regulation
    elapsed = (period - 1) * 12 * 60 + (12 * 60 - time_remaining_period_sec)
    return max(0.0, total_game - elapsed)


def compute_urgencies(margin: float, T_left: float) -> tuple[float, float]:
    """
    Compute urgency levels for 3PT-oriented plays and quick plays.

    margin: our_score - opp_score (positive if we are leading)
    T_left: total seconds remaining in the game

    Returns (three_urgency, quick_urgency), each in [0, 1].
    """
    # Time pressure: 0 early, 1 very late
    time_pressure = 1.0 - (T_left / 2880.0)
    time_pressure = min(max(time_pressure, 0.0), 1.0)

    # Clip margin to [-20, +20]
    m = max(-20.0, min(20.0, margin))

    # Margin pressure: more pressure when tied or losing, less when up
    if m <= 0:
        # Tied or losing: worse margin => more pressure (up to 1 at -20)
        margin_pressure = -m / 20.0
    else:
        # Slight lead still has some pressure; by +5 or more it goes to 0
        margin_pressure = max(0.0, (5.0 - m) / 5.0)

    # Three-urgency: mostly driven by scoreboard pressure, with a small time effect
    three_urgency = 0.8 * margin_pressure + 0.2 * time_pressure
    three_urgency = min(max(three_urgency, 0.0), 1.0)

    # Quick-urgency: mostly driven by time pressure,
    # reduced if we are up by a lot
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

    margin: our_score - opp_score (positive if we are leading)
    period: 1–4
    time_remaining_period_sec: seconds left in current period (0–720)
    """
    # Base matchup scores (no context yet)
    df = build_ml_matchup_table(
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        w_off=w_off,
        w_def=w_def,
    )

    # Add flags for 3PT / quick / slow-2 play types
    df = add_playtype_flags(df)

    # Compute total time left and urgency levels
    T_left = total_time_remaining(period, time_remaining_period_sec)
    three_urgency, quick_urgency = compute_urgencies(margin, T_left)

    # Base score from the ML matchup model
    base = df["PPP_PRED_ML"].to_numpy(dtype=float)

    # Context bonus:
    #  - up to alpha_three PPP for 3PT plays when three_urgency = 1
    #  - up to alpha_quick PPP for quick plays when quick_urgency = 1
    context_boost = (
        alpha_three * three_urgency * df["THREE_PT_PRIORITY"].to_numpy(dtype=float)
        + alpha_quick * quick_urgency * df["QUICK_PRIORITY"].to_numpy(dtype=float)
    )

    # Penalty for "slow 2" plays when urgency is high
    slow_mask = df["IS_SLOW_2"].to_numpy(dtype=float)

    # Combine urgencies for the penalty term (heavier weight on quick_urgency)
    combined_urgency = 0.7 * three_urgency + 1.0 * quick_urgency
    penalty = alpha_penalty * combined_urgency * slow_mask

    df["CONTEXT_SCORE"] = base + context_boost - penalty

    # Explanation string per play type
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
