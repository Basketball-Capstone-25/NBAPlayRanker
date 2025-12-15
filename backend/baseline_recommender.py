import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# Stats that are averaged using possessions (POSS) as weights.
#
# Idea:
# - A guy with 5 possessions shouldn't count the same as a guy with 200.
# - We weight by POSS so high-volume lines have more influence.
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

    What it does:
      - Starts from player-level Synergy stats.
      - Groups them up to a team-level view by season, play type, and side.
      - Computes weighted averages and totals for each group.

    How it works:
      - Create a SIDE column (offense/defense) from TYPE_GROUPING.
      - Group by season + team + play type + side.
      - For each group, sum up possessions and use them as weights when
        averaging PPP and shooting stats.
    """
    df = raw_df.copy()

    # Map Synergy's TYPE_GROUPING to a simple SIDE flag ("offense" / "defense").
    df["SIDE"] = df["TYPE_GROUPING"].str.lower().map(
        {"offensive": "offense", "defensive": "defense"}
    )

    group_cols = ["SEASON", "TEAM_ABBREVIATION", "TEAM_NAME", "PLAY_TYPE", "SIDE"]

    def agg_func(group: pd.DataFrame) -> pd.Series:
        # Total possessions and possession share for this team/playtype/side.
        poss = group["POSS"].sum()
        poss_pct = group["POSS_PCT"].sum()

        # Start with basic totals.
        out = {
            "GP": group["GP"].sum(),      # total games played (sum over players)
            "POSS": poss,                # total possessions for this team/playtype/side
            "POSS_PCT": poss_pct,        # share of team possessions
        }

        # Weighted averages for PPP and related stats (weights = POSS).
        # If poss is zero, we just store NaN to avoid division errors.
        for col in WEIGHT_COLS:
            if poss > 0:
                out[col] = np.average(group[col], weights=group["POSS"])
            else:
                out[col] = np.nan

        # Simple sums for raw scoring and shot attempts.
        out["PTS"] = group["PTS"].sum()
        out["FGM"] = group["FGM"].sum()
        out["FGA"] = group["FGA"].sum()
        return pd.Series(out)

    # Group by team + season + play type + side and apply our aggregator.
    team_df = df.groupby(group_cols, as_index=False).apply(agg_func)
    return team_df


def add_team_reliability_weights(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RELIABILITY_WEIGHT in [0, 1] based on log(POSS).

    What it does:
      - Adds a column that says how much we trust each team/playtype row.

    How it works:
      - Take log1p(POSS) so more possessions => higher weight, but with
        diminishing returns.
      - Normalize by the maximum log value so everything lands in [0, 1].
    """
    result = team_df.copy()
    max_log = np.log1p(result["POSS"]).max()
    result["RELIABILITY_WEIGHT"] = np.log1p(result["POSS"]) / max_log
    return result


def build_league_averages(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build league-average stats per (SEASON, PLAY_TYPE, SIDE).

    LEAGUE_POSS = total possessions for that season/playtype/side.

    What it does:
      - Aggregates across all teams to get a league baseline for each season,
        play type, and side (offense/defense).

    How it works:
      - Group the team_df by (SEASON, PLAY_TYPE, SIDE).
      - Sum up LEAGUE_POSS.
      - Compute weighted averages for PPP and related stats using POSS again.
      - Add a RELIABILITY_WEIGHT for league-level numbers, same log1p idea.
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

    What it does:
      - Runs the full pipeline from raw player rows to:
          1) team_df: team/playtype/side level
          2) league_df: league baselines by season/playtype/side
    """
    # 1) Aggregate player-level Synergy rows up to team level.
    team_df = build_team_playtype_tables(raw_df)

    # 2) Add reliability weights for each team/playtype row.
    team_df = add_team_reliability_weights(team_df)

    # 3) Build league-level baselines.
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

    Returns:
      - A DataFrame with the top-k play types and matchup metrics.

    High-level idea:
      - Take our offensive PPP and their defensive PPP.
      - Shrink both toward league averages depending on reliability.
      - Combine the two into a matchup score (PPP_PRED).
      - Sort and return the top-k play types.
    """
    # ----- Basic validation of inputs -----
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

    # ----- Our offensive profile by play type -----
    #
    # All rows for this season, this team, and SIDE == "offense".
    off = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @our_team and SIDE == 'offense'"
    ).copy()

    # ----- Opponent defensive profile by play type -----
    #
    # All rows for this season, opponent, and SIDE == "defense".
    deff = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @opp_team and SIDE == 'defense'"
    ).copy()

    if off.empty or deff.empty:
        raise ValueError("No data for this matchup (offense or defense table is empty).")

    # ----- League offense baseline per play type -----
    #
    # We pull league-wide offensive PPP and its reliability weight.
    league_off = league_df.query(
        "SEASON == @season and SIDE == 'offense'"
    )[["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]].rename(
        columns={"PPP": "PPP_LEAGUE_OFF", "RELIABILITY_WEIGHT": "REL_OFF_LEAGUE"}
    )

    # ----- League defense baseline per play type -----
    #
    # Same thing but for league-wide defense.
    league_def = league_df.query(
        "SEASON == @season and SIDE == 'defense'"
    )[["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]].rename(
        columns={"PPP": "PPP_LEAGUE_DEF", "RELIABILITY_WEIGHT": "REL_DEF_LEAGUE"}
    )

    # Subset of opponent defense columns that we actually use in the matchup.
    # This keeps the merge clean and focused.
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

    # ----- Join our offense + their defense on play type -----
    #
    # After this, we have columns like:
    #   PPP_OFF, PPP_DEF, POSS_OFF, POSS_DEF, etc.
    merged = off.merge(
        deff_subset,
        on="PLAY_TYPE",
        suffixes=("_OFF", "_DEF"),
    )

    # The offensive FG% came from "off" before the merge, but its name
    # stayed "FG_PCT". Make that explicit.
    merged = merged.rename(columns={"FG_PCT": "FG_PCT_OFF"})

    # Add league offense/defense stats for each play type.
    merged = merged.merge(league_off, on="PLAY_TYPE", how="left")
    merged = merged.merge(league_def, on="PLAY_TYPE", how="left")

    # Reliability weights for our offense and their defense.
    # These come from the team-level tables.
    rel_off = merged["RELIABILITY_WEIGHT_OFF"]
    rel_def = merged["RELIABILITY_WEIGHT_DEF"]

    # ----- Shrink our offense PPP toward league offense PPP -----
    #
    # PPP_OFF_SHRUNK = rel_off * PPP_OFF + (1 - rel_off) * PPP_LEAGUE_OFF
    #
    # Intuition:
    #   - If we have lots of possessions, rel_off is close to 1 and we trust
    #     our team-specific PPP more.
    #   - If we have very few possessions, rel_off is small and we lean on
    #     league PPP to avoid overreacting to small samples.
    merged["PPP_OFF_SHRUNK"] = rel_off * merged["PPP_OFF"] + (1 - rel_off) * merged["PPP_LEAGUE_OFF"]

    # ----- Shrink their defense PPP toward league defense PPP -----
    #
    # Same idea but for opponent defensive PPP.
    merged["PPP_DEF_SHRUNK"] = rel_def * merged["PPP_DEF"] + (1 - rel_def) * merged["PPP_LEAGUE_DEF"]

    # ----- Matchup score: PPP_PRED -----
    #
    # We combine offense and defense into a single prediction:
    #
    #   PPP_PRED = w_off * PPP_OFF_SHRUNK
    #              + w_def * (2 * PPP_LEAGUE_OFF - PPP_DEF_SHRUNK)
    #
    # The second term says:
    #   - if their defense is worse than league, we expect more upside;
    #   - if their defense is better than league, we expect less.
    merged["PPP_PRED"] = (
        w_off * merged["PPP_OFF_SHRUNK"]
        + w_def * (2 * merged["PPP_LEAGUE_OFF"] - merged["PPP_DEF_SHRUNK"])
    )

    # Simple gap: our shrunk PPP minus their shrunk allowed PPP.
    # This is easier to explain to a coach than the full PPP_PRED formula.
    merged["PPP_GAP"] = merged["PPP_OFF_SHRUNK"] - merged["PPP_DEF_SHRUNK"]

    # Sort by predicted value (best plays first),
    # then by offensive possessions as a tiebreaker (more volume first).
    merged = merged.sort_values(["PPP_PRED", "POSS_OFF"], ascending=[False, False])

    # ----- Build a short rationale string for the UI -----
    #
    # This gives the coach a quick English summary of why a play is ranked
    # where it is.
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

    # Columns we return to the API/frontend.
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

    What it does:
      - Loads the Synergy CSV one time.
      - Builds the team_df and league_df tables once.
      - Exposes a `rank()` method the API can call for any matchup.
    """

    def __init__(self, synergy_csv_path: str):
        # Make sure the CSV actually exists so we fail fast if the path is wrong.
        synergy_csv_path = Path(synergy_csv_path)
        if not synergy_csv_path.exists():
            raise FileNotFoundError(synergy_csv_path)

        # Read the raw Synergy CSV into memory.
        self.raw_df = pd.read_csv(synergy_csv_path)

        # Run the full preparation pipeline once at startup.
        self.team_df, self.league_df = prepare_baseline_tables(self.raw_df)

    def rank(self, season: str, our_team: str, opp_team: str, k: int = 5) -> pd.DataFrame:
        """Return top-k play types for a matchup using the baseline pipeline."""
        return rank_playtypes_baseline(self.team_df, self.league_df, season, our_team, opp_team, k=k)


if __name__ == "__main__":
    # Manual smoke test: run this file directly to see a sample ranking.
    # This is handy when developing or debugging.
    csv_path = "data/synergy_playtypes_2019_2025_players.csv"
    rec = BaselineRecommender(csv_path)
    df_top = rec.rank("2019-20", "LAL", "BOS", k=5)
    print(df_top)
