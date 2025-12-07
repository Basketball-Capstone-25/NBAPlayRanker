import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# Stats that are averaged using possessions (POSS) as weights
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
    """
    Aggregate Synergy player-level data to team/season/playtype offense/defense tables.

    Returns one row per (SEASON, TEAM_ABBREVIATION, TEAM_NAME, PLAY_TYPE, SIDE).
    """
    df = raw_df.copy()

    # Map Synergy's TYPE_GROUPING to a simple SIDE flag
    df["SIDE"] = df["TYPE_GROUPING"].str.lower().map(
        {"offensive": "offense", "defensive": "defense"}
    )

    group_cols = ["SEASON", "TEAM_ABBREVIATION", "TEAM_NAME", "PLAY_TYPE", "SIDE"]

    def agg_func(group: pd.DataFrame) -> pd.Series:
        poss = group["POSS"].sum()
        poss_pct = group["POSS_PCT"].sum()

        out = {
            "GP": group["GP"].sum(),      # total games played (sum over players)
            "POSS": poss,                # total possessions for this team/playtype/side
            "POSS_PCT": poss_pct,        # share of team possessions
        }

        # Weighted averages for PPP and related stats (weights = POSS)
        for col in WEIGHT_COLS:
            if poss > 0:
                out[col] = np.average(group[col], weights=group["POSS"])
            else:
                out[col] = np.nan

        # Simple sums for scoring and shots
        out["PTS"] = group["PTS"].sum()
        out["FGM"] = group["FGM"].sum()
        out["FGA"] = group["FGA"].sum()
        return pd.Series(out)

    team_df = df.groupby(group_cols, as_index=False).apply(agg_func)
    return team_df


def add_team_reliability_weights(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RELIABILITY_WEIGHT in [0, 1] based on log(POSS).

    More possessions => higher reliability, with diminishing returns via log1p.
    """
    result = team_df.copy()
    max_log = np.log1p(result["POSS"]).max()
    result["RELIABILITY_WEIGHT"] = np.log1p(result["POSS"]) / max_log
    return result


def build_league_averages(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build league-average stats per (SEASON, PLAY_TYPE, SIDE).

    LEAGUE_POSS = total possessions for that season/playtype/side.
    Other columns are weighted by POSS and get their own RELIABILITY_WEIGHT.
    """
    group_cols = ["SEASON", "PLAY_TYPE", "SIDE"]

    def agg(group: pd.DataFrame) -> pd.Series:
        poss = group["POSS"].sum()
        out = {"LEAGUE_POSS": poss}
        for col in WEIGHT_COLS:
            if poss > 0:
                out[col] = np.average(group[col], weights=group["POSS"])
            else:
                out[col] = np.nan
        return pd.Series(out)

    league_df = team_df.groupby(group_cols, as_index=False).apply(agg)

    max_log = np.log1p(league_df["LEAGUE_POSS"]).max()
    league_df["RELIABILITY_WEIGHT"] = np.log1p(league_df["LEAGUE_POSS"]) / max_log
    return league_df


def prepare_baseline_tables(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build team-level and league-level tables used by the baseline model.
    """
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
    """
    Rank play types for a matchup using the baseline model.

    Uses:
      - our team offense by play type
      - opponent defense by play type
      - league offense/defense baselines

    Returns a DataFrame with the top-k play types and matchup metrics.
    """
    # Basic validation
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

    # Our offensive profile by play type
    off = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @our_team and SIDE == 'offense'"
    ).copy()

    # Opponent defensive profile by play type
    deff = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @opp_team and SIDE == 'defense'"
    ).copy()

    if off.empty or deff.empty:
        raise ValueError("No data for this matchup (offense or defense table is empty).")

    # League offense baseline per play type
    league_off = league_df.query(
        "SEASON == @season and SIDE == 'offense'"
    )[["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]].rename(
        columns={"PPP": "PPP_LEAGUE_OFF", "RELIABILITY_WEIGHT": "REL_OFF_LEAGUE"}
    )

    # League defense baseline per play type
    league_def = league_df.query(
        "SEASON == @season and SIDE == 'defense'"
    )[["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]].rename(
        columns={"PPP": "PPP_LEAGUE_DEF", "RELIABILITY_WEIGHT": "REL_DEF_LEAGUE"}
    )

    # Opponent defense columns used in the matchup
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

    # Join our offense + their defense on play type
    merged = off.merge(
        deff_subset,
        on="PLAY_TYPE",
        suffixes=("_OFF", "_DEF"),
    )

    # Make offense FG% explicit (it came from 'off' before the merge)
    merged = merged.rename(columns={"FG_PCT": "FG_PCT_OFF"})

    # Add league offense/defense stats
    merged = merged.merge(league_off, on="PLAY_TYPE", how="left")
    merged = merged.merge(league_def, on="PLAY_TYPE", how="left")

    # Reliability weights for our offense and their defense
    rel_off = merged["RELIABILITY_WEIGHT_OFF"]
    rel_def = merged["RELIABILITY_WEIGHT_DEF"]

    # Shrink our offense PPP toward league offense PPP
    merged["PPP_OFF_SHRUNK"] = rel_off * merged["PPP_OFF"] + (1 - rel_off) * merged["PPP_LEAGUE_OFF"]

    # Shrink their defense PPP toward league defense PPP
    merged["PPP_DEF_SHRUNK"] = rel_def * merged["PPP_DEF"] + (1 - rel_def) * merged["PPP_LEAGUE_DEF"]

    # Matchup score: higher is better for our offense
    merged["PPP_PRED"] = (
        w_off * merged["PPP_OFF_SHRUNK"]
        + w_def * (2 * merged["PPP_LEAGUE_OFF"] - merged["PPP_DEF_SHRUNK"])
    )

    # Simple gap: our shrunk PPP minus their shrunk allowed PPP
    merged["PPP_GAP"] = merged["PPP_OFF_SHRUNK"] - merged["PPP_DEF_SHRUNK"]

    # Sort by predicted value, then by offensive possessions as tiebreaker
    merged = merged.sort_values(["PPP_PRED", "POSS_OFF"], ascending=[False, False])

    # Build a short rationale string for the UI
    def build_rationale(row: pd.Series) -> str:
        gap = row["PPP_GAP"]
        gap_str = f"+{gap:.3f}" if gap >= 0 else f"{gap:.3f}"
        delta = row["PPP_PRED"] - row["PPP_LEAGUE_OFF"]
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        return (
            f"{row['PLAY_TYPE']}: Pred {delta_str} PPP vs league; "
            f"our {row['PPP_OFF_SHRUNK']:.3f} vs their {row['PPP_DEF_SHRUNK']:.3f} allowed ({gap_str} gap)."
        )

    merged["RATIONALE"] = merged.apply(build_rationale, axis=1)

    cols = [
        "PLAY_TYPE",
        "PPP_PRED",
        "PPP_OFF_SHRUNK",
        "PPP_DEF_SHRUNK",
        "PPP_GAP",
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

    return merged.head(k)[cols].reset_index(drop=True)


class BaselineRecommender:
    """
    Wrapper around the baseline tables and ranking function.

    Loads the CSV once, builds team_df/league_df, and exposes rank().
    """

    def __init__(self, synergy_csv_path: str):
        synergy_csv_path = Path(synergy_csv_path)
        if not synergy_csv_path.exists():
            raise FileNotFoundError(synergy_csv_path)

        self.raw_df = pd.read_csv(synergy_csv_path)
        self.team_df, self.league_df = prepare_baseline_tables(self.raw_df)

    def rank(self, season: str, our_team: str, opp_team: str, k: int = 5) -> pd.DataFrame:
        """Return top-k play types for a matchup."""
        return rank_playtypes_baseline(self.team_df, self.league_df, season, our_team, opp_team, k=k)


if __name__ == "__main__":
    # Manual smoke test: run this file directly to see a sample ranking
    csv_path = "data/synergy_playtypes_2019_2025_players.csv"
    rec = BaselineRecommender(csv_path)
    df_top = rec.rank("2019-20", "LAL", "BOS", k=5)
    print(df_top)
