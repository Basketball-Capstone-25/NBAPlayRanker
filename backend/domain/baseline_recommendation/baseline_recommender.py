#baseline_recommender.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

from domain.baseline_recommendation.interfaces import IBaselineRecommender

# Stats that are averaged using possessions (POSS) as weights.
#
# Why weight by POSS?
# - Player lines have different sample sizes.
# - A player with 5 possessions should not influence a team average as much as a player with 200.
WEIGHT_COLS = [
    "PPP",
    "FG_PCT",
    "EFG_PCT",
    "SCORE_POSS_PCT",
    "TOV_POSS_PCT",
    "SF_POSS_PCT",
    "FT_POSS_PCT",
    "PLUSONE_POSS_PCT",
]

def build_team_playtype_tables(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Synergy player-level data into TEAM-level play-type rows."""
    df = raw_df.copy()

    # Map Synergy's TYPE_GROUPING to a simple SIDE flag.
    df["SIDE"] = df["TYPE_GROUPING"].str.lower().map(
        {"offensive": "offense", "defensive": "defense"}
    )

    group_cols = ["SEASON", "TEAM_ABBREVIATION", "TEAM_NAME", "PLAY_TYPE", "SIDE"]

    def agg_func(group: pd.DataFrame) -> pd.Series:
        poss = group["POSS"].sum()
        poss_pct = group["POSS_PCT"].sum()

        out = {
            "GP": group["GP"].sum(),       # total games (sum over players; used only as reference)
            "POSS": poss,                 # total possessions for the team/playtype/side
            "POSS_PCT": poss_pct,         # share of team possessions (summing player shares)
        }

        # Weighted averages for rate stats
        for col in WEIGHT_COLS:
            out[col] = np.average(group[col], weights=group["POSS"]) if poss > 0 else np.nan

        # Raw sums for simple context/debug
        out["PTS"] = group["PTS"].sum()
        out["FGM"] = group["FGM"].sum()
        out["FGA"] = group["FGA"].sum()
        return pd.Series(out)

    # Group and aggregate
    team_df = df.groupby(group_cols, as_index=False).apply(agg_func)
    return team_df

def add_team_reliability_weights(team_df: pd.DataFrame) -> pd.DataFrame:
    """Add RELIABILITY_WEIGHT in [0, 1] based on log1p(POSS)."""
    result = team_df.copy()
    max_log = np.log1p(result["POSS"]).max()
    result["RELIABILITY_WEIGHT"] = np.log1p(result["POSS"]) / max_log if max_log > 0 else 0.0
    return result

def build_league_averages(team_df: pd.DataFrame) -> pd.DataFrame:
    """Build league-average stats per (SEASON, PLAY_TYPE, SIDE)."""
    group_cols = ["SEASON", "PLAY_TYPE", "SIDE"]

    def agg(group: pd.DataFrame) -> pd.Series:
        poss = group["POSS"].sum()
        out = {"LEAGUE_POSS": poss}
        for col in WEIGHT_COLS:
            out[col] = np.average(group[col], weights=group["POSS"]) if poss > 0 else np.nan
        return pd.Series(out)

    league_df = team_df.groupby(group_cols, as_index=False).apply(agg)

    max_log = np.log1p(league_df["LEAGUE_POSS"]).max()
    league_df["RELIABILITY_WEIGHT"] = np.log1p(league_df["LEAGUE_POSS"]) / max_log if max_log > 0 else 0.0
    return league_df

def prepare_baseline_tables(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full baseline ETL pipeline:"""
    team_df = build_team_playtype_tables(raw_df)
    team_df = add_team_reliability_weights(team_df)
    league_df = build_league_averages(team_df)
    return team_df, league_df

def rank_playtypes_baseline(
    team_df: pd.DataFrame,
    league_df: pd.DataFrame,
    season: str,
    our_team: str,
    opp_team: str,
    k: int = 5,
    w_off: float = 0.7,
    w_def: float = 0.3,
) -> pd.DataFrame:
    """Rank play types for a matchup using an explainable baseline model."""
    valid_seasons = set(team_df["SEASON"].unique())
    if season not in valid_seasons:
        raise ValueError(f"Unknown season '{season}'. Valid seasons: {sorted(valid_seasons)}")

    valid_teams = set(team_df["TEAM_ABBREVIATION"].unique())
    if our_team not in valid_teams:
        raise ValueError(f"Unknown our_team '{our_team}'.")
    if opp_team not in valid_teams:
        raise ValueError(f"Unknown opp_team '{opp_team}'.")
    if not (1 <= k <= 10):
        raise ValueError("k must be between 1 and 10.")
    off = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @our_team and SIDE == 'offense'"
    ).copy()

    deff = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @opp_team and SIDE == 'defense'"
    ).copy()

    if off.empty or deff.empty:
        raise ValueError("No data for this matchup (offense or defense table is empty).")
    league_off = league_df.query(
        "SEASON == @season and SIDE == 'offense'"
    )[["PLAY_TYPE", "PPP"]].rename(columns={"PPP": "PPP_LEAGUE_OFF"})

    league_def = league_df.query(
        "SEASON == @season and SIDE == 'defense'"
    )[["PLAY_TYPE", "PPP"]].rename(columns={"PPP": "PPP_LEAGUE_DEF"})

    # Keep only the defense columns we need (makes merge cleaner)
    deff_subset = deff[
        [
            "PLAY_TYPE",
            "PPP",
            "POSS",
            "POSS_PCT",
            "RELIABILITY_WEIGHT",
            "FG_PCT",
            "EFG_PCT",
            "SCORE_POSS_PCT",
            "TOV_POSS_PCT",
        ]
    ].copy()
    merged = off.merge(
        deff_subset,
        on="PLAY_TYPE",
        suffixes=("_OFF", "_DEF"),
    )

    # "FG_PCT" from offense becomes FG_PCT_OFF explicitly (the merge won't suffix it
    # if the defense subset also contains FG_PCT — so we force clarity).
    if "FG_PCT" in merged.columns:
        merged = merged.rename(columns={"FG_PCT": "FG_PCT_OFF"})

    # Add league anchors per play type
    merged = merged.merge(league_off, on="PLAY_TYPE", how="left")
    merged = merged.merge(league_def, on="PLAY_TYPE", how="left")
    #
    # PPP_SHRUNK = REL * PPP_TEAM + (1-REL) * PPP_LEAGUE
    rel_off = merged["RELIABILITY_WEIGHT_OFF"]
    rel_def = merged["RELIABILITY_WEIGHT_DEF"]

    merged["PPP_OFF_SHRUNK"] = rel_off * merged["PPP_OFF"] + (1 - rel_off) * merged["PPP_LEAGUE_OFF"]
    merged["PPP_DEF_SHRUNK"] = rel_def * merged["PPP_DEF"] + (1 - rel_def) * merged["PPP_LEAGUE_DEF"]
    #
    # PPP_PRED = w_off * PPP_OFF_SHRUNK
    #          + w_def * (2*PPP_LEAGUE_OFF - PPP_DEF_SHRUNK)
    #
    # Interpret the defense term:
    # - If opponent allows MORE than league (PPP_DEF_SHRUNK high), the term becomes smaller.
    # - If opponent allows LESS than league (PPP_DEF_SHRUNK low), the term becomes larger.
    merged["PPP_PRED"] = (
        float(w_off) * merged["PPP_OFF_SHRUNK"]
        + float(w_def) * (2 * merged["PPP_LEAGUE_OFF"] - merged["PPP_DEF_SHRUNK"])
    )

    # Simple explainable gap (coach-friendly)
    merged["PPP_GAP"] = merged["PPP_OFF_SHRUNK"] - merged["PPP_DEF_SHRUNK"]
    merged = merged.sort_values(["PPP_PRED", "POSS_OFF"], ascending=[False, False])

    def build_rationale(row: pd.Series) -> str:
        gap = float(row["PPP_GAP"])
        gap_str = f"+{gap:.3f}" if gap >= 0 else f"{gap:.3f}"
        rel_off = float(row["RELIABILITY_WEIGHT_OFF"])
        rel_def = float(row["RELIABILITY_WEIGHT_DEF"])
        return (
            f"{row['PLAY_TYPE']}: Pred {row['PPP_PRED']:.3f} PPP. "
            f"Our(off) {row['PPP_OFF_SHRUNK']:.3f} vs Opp(def) {row['PPP_DEF_SHRUNK']:.3f} ({gap_str}). "
            f"Reliability off={rel_off:.2f}, def={rel_def:.2f}."
        )

    merged["RATIONALE"] = merged.apply(build_rationale, axis=1)
    #
    # These columns are used by the updated frontend Matchup/Baseline page to show a full breakdown.
    cols = [
        "PLAY_TYPE",
        "PPP_PRED",
        "PPP_OFF_SHRUNK",
        "PPP_DEF_SHRUNK",
        "PPP_GAP",
        "PPP_LEAGUE_OFF",
        "PPP_LEAGUE_DEF",
        "RELIABILITY_WEIGHT_OFF",
        "RELIABILITY_WEIGHT_DEF",
        "POSS_OFF",
        "POSS_DEF",
        "POSS_PCT_OFF",
        "POSS_PCT_DEF",
        "FG_PCT_OFF",
        "EFG_PCT_OFF",
        "EFG_PCT_DEF",
        "SCORE_POSS_PCT_OFF",
        "SCORE_POSS_PCT_DEF",
        "TOV_POSS_PCT_OFF",
        "TOV_POSS_PCT_DEF",
        "RATIONALE",
    ]

    cols = [c for c in cols if c in merged.columns]
    return merged.head(k)[cols].reset_index(drop=True)

class BaselineRecommender(IBaselineRecommender):
    """Simple wrapper used by the API:"""

    def __init__(self, synergy_csv_path: str):
        synergy_csv_path = Path(synergy_csv_path)
        if not synergy_csv_path.exists():
            raise FileNotFoundError(synergy_csv_path)

        self.raw_df = pd.read_csv(synergy_csv_path)
        self.team_df, self.league_df = prepare_baseline_tables(self.raw_df)

    def rank(self, season: str, our_team: str, opp_team: str, k: int = 5) -> pd.DataFrame:
        return rank_playtypes_baseline(
            self.team_df, self.league_df, season, our_team, opp_team, k=k
        )
