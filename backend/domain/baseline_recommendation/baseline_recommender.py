#baseline_recommender.py

import pandas as pd
from infrastructure.data_access.dataset_metadata import read_versioned_csv
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple

from domain.baseline_recommendation.interfaces import IBaselineRecommender

AGGREGATION_VERSION = "synergy_regular_season_v2"
SEASON_PHASE = "regular_season"

# These possession-based rates overlap. They are not mutually exclusive outcomes.
POSSESSION_RATE_COLS = [
    "SCORE_POSS_PCT", "TOV_POSS_PCT", "SF_POSS_PCT", "FT_POSS_PCT", "PLUSONE_POSS_PCT",
]
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


def regular_season_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Select an identified regular season before counting player-ending plays."""
    required = {
        "SEASON", "SEASON_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
        "PLAY_TYPE", "TYPE_GROUPING", "GP", "POSS", "PTS", "FGM", "FGA", "EFG_PCT",
        *POSSESSION_RATE_COLS,
    }
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"Play-type source is missing columns: {', '.join(missing)}")
    phase = raw_df["SEASON_ID"].astype(str)
    frame = raw_df.loc[phase.str.fullmatch(r"2\d{4}")].copy()
    if frame.empty:
        raise ValueError("Play-type source has no regular-season rows.")
    years = frame["SEASON_ID"].astype(str).str[1:].astype(int)
    labels = years.map(lambda year: f"{year}-{(year + 1) % 100:02d}")
    if not frame["SEASON"].eq(labels).all():
        raise ValueError("SEASON and regular-season SEASON_ID must agree.")
    keys = ["SEASON", "PLAYER_ID", "TEAM_ABBREVIATION", "PLAY_TYPE", "TYPE_GROUPING"]
    if frame[keys].isna().any().any():
        raise ValueError("Play-type source requires known player, team, season and play-type keys.")
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate player / team / play-type rows would double-count possessions.")
    if "ENTITY_TYPE" in frame and not frame["ENTITY_TYPE"].eq("Player").all():
        raise ValueError("Play-type aggregation requires player rows without team-total duplicates.")
    if not frame["TYPE_GROUPING"].isin(["Offensive", "Defensive"]).all():
        raise ValueError("Play-type source contains an unknown offense/defense grouping.")
    counts = frame[["GP", "POSS", "PTS", "FGM", "FGA"]].to_numpy(dtype=float)
    rates = frame[["EFG_PCT", *POSSESSION_RATE_COLS]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or not np.isfinite(rates).all():
        raise ValueError("Play-type counts and rates must be finite.")
    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all() or (frame["POSS"] <= 0).any() or (frame["FGM"] > frame["FGA"]).any():
        raise ValueError("Play-type counts must be valid and possessions positive.")
    if (rates < 0).any() or (rates[:, 0] > 1.5).any() or (rates[:, 1:] > 1).any():
        raise ValueError("Play-type source contains an invalid rate.")
    return frame


def _aggregate_rates(group: pd.DataFrame) -> dict[str, float]:
    """Keep each rate's denominator; rounded eFG is weighted by attempts."""
    poss = float(group["POSS"].sum())
    fga = float(group["FGA"].sum())
    out = {
        "PPP": float(group["PTS"].sum()) / poss if poss > 0 else np.nan,
        "FG_PCT": float(group["FGM"].sum()) / fga if fga > 0 else np.nan,
        "EFG_PCT": float((group["EFG_PCT"] * group["FGA"]).sum()) / fga if fga > 0 else np.nan,
    }
    for column in POSSESSION_RATE_COLS:
        out[column] = float((group[column] * group["POSS"]).sum()) / poss if poss > 0 else np.nan
    return out


def build_team_playtype_tables(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate regular-season player rows without mixing playoff samples."""
    df = regular_season_rows(raw_df)

    # Map Synergy's TYPE_GROUPING to a simple SIDE flag.
    df["SIDE"] = df["TYPE_GROUPING"].str.lower().map(
        {"offensive": "offense", "defensive": "defense"}
    )

    group_cols = ["SEASON", "TEAM_ABBREVIATION", "TEAM_NAME", "PLAY_TYPE", "SIDE"]

    def agg_func(group: pd.DataFrame) -> pd.Series:
        poss = group["POSS"].sum()

        out = {
            # Retained for older exports: this is player appearances, NOT team games.
            "GP": group["GP"].sum(),
            "POSS": poss,                 # total possessions for the team/playtype/side
            "POSS_PCT": np.nan,           # placeholder — recomputed after groupby
        }

        out.update(_aggregate_rates(group))

        # Raw sums for simple context/debug
        out["PTS"] = group["PTS"].sum()
        out["FGM"] = group["FGM"].sum()
        out["FGA"] = group["FGA"].sum()
        return pd.Series(out)

    # Group and aggregate
    team_df = df.groupby(group_cols, as_index=False).apply(agg_func)

    # Recompute POSS_PCT as the fraction of a team's total possessions
    # that go to each play type (per season/team/side).
    team_totals = team_df.groupby(["SEASON", "TEAM_ABBREVIATION", "SIDE"])["POSS"].transform("sum")
    team_df["POSS_PCT"] = team_df["POSS"] / team_totals
    team_df.attrs.update(raw_df.attrs)
    team_df.attrs.update(aggregation_version=AGGREGATION_VERSION, season_phase=SEASON_PHASE)

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
        out = {"LEAGUE_POSS": poss, "PTS": group["PTS"].sum(), "FGA": group["FGA"].sum()}
        out.update(_aggregate_rates(group))
        return pd.Series(out)

    league_df = team_df.groupby(group_cols, as_index=False).apply(agg)

    max_log = np.log1p(league_df["LEAGUE_POSS"]).max()
    league_df["RELIABILITY_WEIGHT"] = np.log1p(league_df["LEAGUE_POSS"]) / max_log if max_log > 0 else 0.0
    league_df.attrs.update(team_df.attrs)
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
    if our_team == opp_team:
        raise ValueError("Our team and opponent must be different.")
    if not np.isfinite([w_off, w_def]).all() or min(w_off, w_def) < 0 or not np.isclose(w_off + w_def, 1):
        raise ValueError("Offense and defense weights must be nonnegative and sum to one.")
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
    # A defense that allows more points should raise, not lower, the offensive score.
    # This weighted historical estimate is not a validated pre-game forecast.
    merged["PPP_PRED"] = (
        float(w_off) * merged["PPP_OFF_SHRUNK"]
        + float(w_def) * merged["PPP_DEF_SHRUNK"]
    )

    # A descriptive difference between historical profiles, not a matchup advantage.
    merged["PPP_GAP"] = merged["PPP_OFF_SHRUNK"] - merged["PPP_DEF_SHRUNK"]
    merged = merged.sort_values(["PPP_PRED", "POSS_OFF"], ascending=[False, False])

    def build_rationale(row: pd.Series) -> str:
        gap = float(row["PPP_GAP"])
        gap_str = f"+{gap:.3f}" if gap >= 0 else f"{gap:.3f}"
        rel_off = float(row["RELIABILITY_WEIGHT_OFF"])
        rel_def = float(row["RELIABILITY_WEIGHT_DEF"])
        return (
            f"{row['PLAY_TYPE']}: Pred {row['PPP_PRED']:.3f} PPP. "
            f"Our(off) {row['PPP_OFF_SHRUNK']:.3f} vs Opp(allowed) {row['PPP_DEF_SHRUNK']:.3f} ({gap_str}). "
            f"Reliability off={rel_off:.2f}, def={rel_def:.2f}. "
            "Regular-season historical estimate. "
            "Game context was not used: score margin, period and clock do not affect this baseline ranking."
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

        self.raw_df = read_versioned_csv(synergy_csv_path)
        self.team_df, self.league_df = prepare_baseline_tables(self.raw_df)

    def rank(self, season: str, our_team: str, opp_team: str, k: int = 5) -> pd.DataFrame:
        return rank_playtypes_baseline(
            self.team_df, self.league_df, season, our_team, opp_team, k=k
        )
