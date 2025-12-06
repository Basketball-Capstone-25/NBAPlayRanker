import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

# ---------- Data prep ----------

# These are the numeric columns where we care about weighted averages.
# We will average these using possessions (POSS) as weights.
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
    Aggregate Synergy player-level data to team-season-playtype offense/defense tables.

    Input:
        raw_df: player-level Synergy data (one row per player, play type, season, etc.)

    Output:
        team_df: one row per (season, team, play-type, side [offense/defense]) with
                 total possessions, weighted PPP, and other weighted stats.
    """
    df = raw_df.copy()

    # Map Synergy's TYPE_GROUPING to a simpler SIDE flag: "offense" or "defense".
    df["SIDE"] = df["TYPE_GROUPING"].str.lower().map({"offensive": "offense", "defensive": "defense"})

    # We want team-season-playtype-level aggregates, split by offense vs defense.
    group_cols = ["SEASON", "TEAM_ABBREVIATION", "TEAM_NAME", "PLAY_TYPE", "SIDE"]

    def agg_func(group: pd.DataFrame) -> pd.Series:
        # Total possessions and possession share.
        poss = group["POSS"].sum()
        poss_pct = group["POSS_PCT"].sum()
        out = {
            "GP": group["GP"].sum(),     # total games played (sum across players)
            "POSS": poss,               # total possessions for this team+playtype+side
            "POSS_PCT": poss_pct,       # share of team possessions
        }

        # Weighted averages for PPP and related shooting/turnover stats.
        # We use POSS as weights, so high-usage players influence more.
        for col in WEIGHT_COLS:
            if poss > 0:
                out[col] = np.average(group[col], weights=group["POSS"])
            else:
                out[col] = np.nan

        # Simple sums for scoring/shot counts.
        out["PTS"] = group["PTS"].sum()
        out["FGM"] = group["FGM"].sum()
        out["FGA"] = group["FGA"].sum()
        return pd.Series(out)

    # Apply the aggregation for each team-season-playtype-side.
    team_df = df.groupby(group_cols, as_index=False).apply(agg_func)
    return team_df


def add_team_reliability_weights(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 0–1 reliability weight based on log(POSS) (what you described in Phase A).

    Idea:
        - More possessions → more reliable stat.
        - Use log(1 + POSS) so the weight grows but with diminishing returns.
        - Normalize by the maximum value so the largest sample gets weight 1.0.
    """
    result = team_df.copy()
    max_log = np.log1p(result["POSS"]).max()
    result["RELIABILITY_WEIGHT"] = np.log1p(result["POSS"]) / max_log
    return result


def build_league_averages(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build league-average PPP/etc per season / play_type / side with a reliability weight.

    We aggregate across *all* teams to get the league-wide baseline for each play type.
    """
    group_cols = ["SEASON", "PLAY_TYPE", "SIDE"]

    def agg(group: pd.DataFrame) -> pd.Series:
        poss = group["POSS"].sum()
        out = {"LEAGUE_POSS": poss}
        # Weighted league averages for PPP and related stats.
        for col in WEIGHT_COLS:
            if poss > 0:
                out[col] = np.average(group[col], weights=group["POSS"])
            else:
                out[col] = np.nan
        return pd.Series(out)

    league_df = team_df.groupby(group_cols, as_index=False).apply(agg)
    # Reliability weight for league-level stats, again based on log possessions.
    max_log = np.log1p(league_df["LEAGUE_POSS"]).max()
    league_df["RELIABILITY_WEIGHT"] = np.log1p(league_df["LEAGUE_POSS"]) / max_log
    return league_df


def prepare_baseline_tables(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full Phase-A prep: team tables + league averages + reliability weights.

    Steps:
        1. Aggregate player-level Synergy data to team-level offense/defense tables.
        2. Add team-level reliability weights.
        3. Build league-average stats and league reliability weights.
    """
    team_df = build_team_playtype_tables(raw_df)
    team_df = add_team_reliability_weights(team_df)
    league_df = build_league_averages(team_df)
    return team_df, league_df


# ---------- Baseline ranking ----------

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
    Baseline Top-K ranking for a given matchup, as in the Capstone EOI.

    Concept:
        - Uses team offense vs opponent defense for each play type.
        - Shrinks both teams' PPP toward league averages using reliability weights.
        - Combines them into a matchup-specific predicted PPP score.

    Score rule (higher is better for us):
        PPP_PRED = w_off * PPP_off_shrunk
                   + w_def * (2 * PPP_league_off - PPP_def_shrunk)

    where:
        - PPP_off_shrunk  = our offense PPP blended with league-offense PPP
        - PPP_def_shrunk  = their defense PPP blended with league-defense PPP
        - PPP_league_off  = overall league-offense PPP for that play type
    """

    # --- validation of inputs ---
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

    # --- slice offense / defense for the matchup ---
    # Our offensive profile by play type for the given season.
    off = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @our_team and SIDE == 'offense'"
    ).copy()

    # Opponent defensive profile by play type for the given season.
    deff = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @opp_team and SIDE == 'defense'"
    ).copy()

    if off.empty or deff.empty:
        raise ValueError("No data for this matchup (offense or defense table is empty).")

    # League offensive baseline for each play type in this season.
    league_off = league_df.query(
        "SEASON == @season and SIDE == 'offense'"
    )[["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]].rename(
        columns={"PPP": "PPP_LEAGUE_OFF", "RELIABILITY_WEIGHT": "REL_OFF_LEAGUE"}
    )

    # League defensive baseline for each play type in this season.
    league_def = league_df.query(
        "SEASON == @season and SIDE == 'defense'"
    )[["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]].rename(
        columns={"PPP": "PPP_LEAGUE_DEF", "RELIABILITY_WEIGHT": "REL_DEF_LEAGUE"}
    )

    # Select only the defensive columns we need from the opponent.
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

    # Join our offense and their defense on PLAY_TYPE.
    merged = off.merge(
        deff_subset,
        on="PLAY_TYPE",
        suffixes=("_OFF", "_DEF"),
    )

    # Make offense FG% explicit (it came in from 'off' and now has suffixes).
    merged = merged.rename(columns={"FG_PCT": "FG_PCT_OFF"})

    # Add league offense/defense rows per play type.
    merged = merged.merge(league_off, on="PLAY_TYPE", how="left")
    merged = merged.merge(league_def, on="PLAY_TYPE", how="left")

    # Reliability weights for our offense and their defense.
    rel_off = merged["RELIABILITY_WEIGHT_OFF"]
    rel_def = merged["RELIABILITY_WEIGHT_DEF"]

    # Shrink our offense PPP toward league offense PPP if we don't have many possessions.
    merged["PPP_OFF_SHRUNK"] = rel_off * merged["PPP_OFF"] + (1 - rel_off) * merged["PPP_LEAGUE_OFF"]

    # Shrink their defense PPP toward league defense PPP if they don't have many possessions.
    merged["PPP_DEF_SHRUNK"] = rel_def * merged["PPP_DEF"] + (1 - rel_def) * merged["PPP_LEAGUE_DEF"]

    # Score rule (higher = better for us).
    # We prefer:
    #   - high offense PPP for us
    #   - low defense PPP for them (compared to league baseline)
    merged["PPP_PRED"] = (
        w_off * merged["PPP_OFF_SHRUNK"]
        + w_def * (2 * merged["PPP_LEAGUE_OFF"] - merged["PPP_DEF_SHRUNK"])
    )

    # Gap metric = our shrunk PPP minus their shrunk allowed PPP (simple interpretation).
    merged["PPP_GAP"] = merged["PPP_OFF_SHRUNK"] - merged["PPP_DEF_SHRUNK"]

    # Sort by predicted value, then by offensive possessions as a tiebreaker.
    merged = merged.sort_values(["PPP_PRED", "POSS_OFF"], ascending=[False, False])

    # Build a short human-readable rationale string for each row (used in the UI).
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

    # Final set of columns returned to the API / frontend.
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

    # Return Top-K rows (sorted above), with a clean index.
    return merged.head(k)[cols].reset_index(drop=True)


class BaselineRecommender:
    """
    Thin wrapper so FastAPI / notebooks can share the same logic.

    This class:
        - loads the Synergy CSV once,
        - precomputes the team and league tables,
        - exposes a simple rank(...) method used by the API and tests.
    """

    def __init__(self, synergy_csv_path: str):
        synergy_csv_path = Path(synergy_csv_path)
        if not synergy_csv_path.exists():
            raise FileNotFoundError(synergy_csv_path)

        # Read the raw Synergy data once.
        self.raw_df = pd.read_csv(synergy_csv_path)

        # Precompute team-level and league-level tables used by the baseline model.
        self.team_df, self.league_df = prepare_baseline_tables(self.raw_df)

    def rank(self, season: str, our_team: str, opp_team: str, k: int = 5) -> pd.DataFrame:
        """
        Public helper used by FastAPI and tests.

        Given a season and two team codes, return the Top-K play types for that matchup
        using the baseline ranking logic.
        """
        return rank_playtypes_baseline(self.team_df, self.league_df, season, our_team, opp_team, k=k)


if __name__ == "__main__":
    # Quick manual smoke test (you can run: python baseline_recommender.py)
    csv_path = "data/synergy_playtypes_2019_2025_players.csv"
    rec = BaselineRecommender(csv_path)
    df_top = rec.rank("2019-20", "LAL", "BOS", k=5)
    print(df_top)
