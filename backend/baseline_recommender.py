"""backend/baseline_recommender.py

Baseline (non-ML) recommender used in the PSPI.

What this file demonstrates:
1) A real, reproducible data pipeline:
   - load the raw Synergy snapshot (player-level),
   - aggregate to team-level play-type stats (possession-weighted),
   - compute league averages,
   - apply shrinkage to reduce small-sample noise,
   - rank play types for a given matchup.

2) An explainable formula you can defend:

   PPP_PRED = w_off * PPP_OFF_SHRUNK + w_def * PPP_DEF_SHRUNK

   where shrinkage is:
   PPP_OFF_SHRUNK = rel_off * PPP_OFF + (1-rel_off) * PPP_LEAGUE_OFF
   PPP_DEF_SHRUNK = rel_def * PPP_DEF + (1-rel_def) * PPP_LEAGUE_DEF

   rel_off/rel_def are reliability weights in [0,1] derived from possessions.

Why we recompute POSS_PCT:
- The raw dataset is player-level, and the Synergy 'POSS_PCT' at player-level is
  not guaranteed to represent team usage. To avoid misleading UI and metrics, we
  recompute team-level POSS_PCT as:

      POSS_PCT = POSS(play_type) / POSS(all play types)   (per team/season/side)

That change makes the Data Explorer more credible and defensible.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# Stats that should be possession-weighted when aggregating player rows to a team row.
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


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Safely cast columns to numeric (NaN on bad values)."""

    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_team_playtype_tables(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level rows into team-level play-type rows.

    Returns one row per:
        (SEASON, TEAM_ABBREVIATION, TEAM_NAME, PLAY_TYPE, SIDE)

    Notes:
    - We use possession-weighted averages for rate statistics.
    - We recompute POSS_PCT at the team-level (see module docstring).
    """

    df = raw_df.copy()

    # Normalize SIDE from TYPE_GROUPING.
    # (If TYPE_GROUPING ever contains unexpected values, drop those rows.)
    df["SIDE"] = (
        df.get("TYPE_GROUPING")
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"offensive": "offense", "defensive": "defense"})
    )
    df = df[df["SIDE"].notna()].copy()

    # Ensure numeric types for aggregation.
    df = _to_numeric(
        df,
        cols=[
            "GP",
            "POSS",
            "PTS",
            "FGM",
            "FGA",
            *WEIGHT_COLS,
        ],
    )

    # Drop rows with no possessions (can't weight anything).
    df = df[df["POSS"].fillna(0) > 0].copy()

    group_cols = ["SEASON", "TEAM_ABBREVIATION", "TEAM_NAME", "PLAY_TYPE", "SIDE"]

    # We build aggregates using index-based groupby to keep the code explicit and
    # avoid the double-counting pitfalls of summing player POSS_PCT.
    idx = df.set_index(group_cols)

    poss = idx["POSS"].groupby(level=group_cols).sum().rename("POSS")
    gp = idx["GP"].groupby(level=group_cols).max().rename("GP")  # max is sensible at team-level
    pts = idx["PTS"].groupby(level=group_cols).sum().rename("PTS")
    fgm = idx["FGM"].groupby(level=group_cols).sum().rename("FGM")
    fga = idx["FGA"].groupby(level=group_cols).sum().rename("FGA")

    out = pd.concat([gp, poss, pts, fgm, fga], axis=1)

    # Possession-weighted averages for rate stats.
    for col in WEIGHT_COLS:
        if col not in idx.columns:
            continue
        num = (idx[col] * idx["POSS"]).groupby(level=group_cols).sum()
        out[col] = (num / poss).replace([np.inf, -np.inf], np.nan)

    out = out.reset_index()

    # Recompute POSS_PCT (team usage rate) from team-level possessions.
    team_total = out.groupby(["SEASON", "TEAM_ABBREVIATION", "SIDE"])["POSS"].transform("sum")
    out["POSS_PCT"] = np.where(team_total > 0, out["POSS"] / team_total, np.nan)

    return out


def add_team_reliability_weights(team_df: pd.DataFrame) -> pd.DataFrame:
    """Add RELIABILITY_WEIGHT in [0,1] based on log1p(POSS).

    Interpretation:
    - If a team only ran a play type a few times, we trust the PPP less.
    - If the team ran it a lot, we trust it more.
    """

    result = team_df.copy()
    max_log = float(np.log1p(result["POSS"]).max()) if len(result) else 0.0
    if max_log <= 0:
        result["RELIABILITY_WEIGHT"] = 0.0
    else:
        result["RELIABILITY_WEIGHT"] = np.log1p(result["POSS"]) / max_log
    return result


def build_league_averages(team_df: pd.DataFrame) -> pd.DataFrame:
    """Build league-average stats per (SEASON, PLAY_TYPE, SIDE).

    We compute league averages from the team-level table so that the baseline is
    consistent: team PPP shrinks toward league PPP built from team rows.
    """

    group_cols = ["SEASON", "PLAY_TYPE", "SIDE"]

    idx = team_df.set_index(group_cols)
    poss = idx["POSS"].groupby(level=group_cols).sum().rename("LEAGUE_POSS")

    out = pd.DataFrame(poss)
    for col in WEIGHT_COLS:
        if col not in idx.columns:
            continue
        num = (idx[col] * idx["POSS"]).groupby(level=group_cols).sum()
        out[col] = (num / poss).replace([np.inf, -np.inf], np.nan)

    out = out.reset_index()

    max_log = float(np.log1p(out["LEAGUE_POSS"]).max()) if len(out) else 0.0
    if max_log <= 0:
        out["RELIABILITY_WEIGHT"] = 0.0
    else:
        out["RELIABILITY_WEIGHT"] = np.log1p(out["LEAGUE_POSS"]) / max_log

    return out


def prepare_baseline_tables(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the team_df and league_df used by the backend."""

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
    """Rank play types for a matchup using the baseline model.

    Baseline formula (after shrinkage):

        PPP_PRED = w_off * PPP_OFF_SHRUNK + w_def * PPP_DEF_SHRUNK

    Returns a DataFrame with extra columns so the frontend can show a
    transparent breakdown during the demo.
    """

    # ----- input validation -----
    valid_seasons = set(team_df["SEASON"].unique())
    if season not in valid_seasons:
        raise ValueError(f"Unknown season '{season}'.")

    valid_teams = set(team_df["TEAM_ABBREVIATION"].unique())
    if our_team not in valid_teams:
        raise ValueError(f"Unknown our_team '{our_team}'.")
    if opp_team not in valid_teams:
        raise ValueError(f"Unknown opp_team '{opp_team}'.")

    if not (1 <= k <= 10):
        raise ValueError("k must be between 1 and 10.")

    if w_off < 0 or w_def < 0 or (w_off + w_def) <= 0:
        raise ValueError("w_off and w_def must be non-negative and not both zero.")

    # Normalize weights so w_off + w_def = 1 (prevents accidental scaling).
    s = float(w_off + w_def)
    w_off = float(w_off / s)
    w_def = float(w_def / s)

    # ----- slice the team tables for this matchup -----
    off = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @our_team and SIDE == 'offense'"
    ).copy()
    deff = team_df.query(
        "SEASON == @season and TEAM_ABBREVIATION == @opp_team and SIDE == 'defense'"
    ).copy()

    if off.empty or deff.empty:
        raise ValueError("No data for this matchup (offense or defense table is empty).")

    # League baselines for this season.
    league_off = league_df.query("SEASON == @season and SIDE == 'offense'")[
        ["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]
    ].rename(columns={"PPP": "PPP_LEAGUE_OFF", "RELIABILITY_WEIGHT": "REL_LEAGUE_OFF"})

    league_def = league_df.query("SEASON == @season and SIDE == 'defense'")[
        ["PLAY_TYPE", "PPP", "RELIABILITY_WEIGHT"]
    ].rename(columns={"PPP": "PPP_LEAGUE_DEF", "RELIABILITY_WEIGHT": "REL_LEAGUE_DEF"})

    # Keep a focused subset of opponent defense features.
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
            "SF_POSS_PCT",
            "FT_POSS_PCT",
            "PLUSONE_POSS_PCT",
        ]
    ].copy()

    merged = off.merge(deff_subset, on="PLAY_TYPE", suffixes=("_OFF", "_DEF"))

    # Join league baselines.
    merged = merged.merge(league_off, on="PLAY_TYPE", how="left")
    merged = merged.merge(league_def, on="PLAY_TYPE", how="left")

    # Reliability weights for offense and defense (team-level).
    rel_off = merged["RELIABILITY_WEIGHT_OFF"].fillna(0.0)
    rel_def = merged["RELIABILITY_WEIGHT_DEF"].fillna(0.0)

    # Shrink offense PPP toward league offense PPP.
    merged["PPP_OFF_SHRUNK"] = rel_off * merged["PPP_OFF"] + (1 - rel_off) * merged[
        "PPP_LEAGUE_OFF"
    ]

    # Shrink defense (allowed PPP) toward league defense PPP.
    merged["PPP_DEF_SHRUNK"] = rel_def * merged["PPP_DEF"] + (1 - rel_def) * merged[
        "PPP_LEAGUE_DEF"
    ]

    # Baseline matchup prediction.
    merged["PPP_PRED"] = w_off * merged["PPP_OFF_SHRUNK"] + w_def * merged["PPP_DEF_SHRUNK"]

    # "Gap" metric: how our predicted PPP compares to what they usually allow.
    merged["PPP_GAP"] = merged["PPP_PRED"] - merged["PPP_DEF_SHRUNK"]

    # Sort and pick top K.
    merged = merged.sort_values(["PPP_PRED", "POSS_OFF"], ascending=[False, False])

    def build_rationale(row: pd.Series) -> str:
        gap = float(row["PPP_GAP"])
        gap_str = f"+{gap:.3f}" if gap >= 0 else f"{gap:.3f}"
        delta_vs_league = float(row["PPP_PRED"] - row["PPP_LEAGUE_OFF"])
        delta_str = f"+{delta_vs_league:.3f}" if delta_vs_league >= 0 else f"{delta_vs_league:.3f}"

        return (
            f"{row['PLAY_TYPE']}: predicted {delta_str} PPP vs league; "
            f"our shrunk={row['PPP_OFF_SHRUNK']:.3f}, opp allowed shrunk={row['PPP_DEF_SHRUNK']:.3f} "
            f"(gap {gap_str})."
        )

    merged["RATIONALE"] = merged.apply(build_rationale, axis=1)

    # Include extra "math" columns so the frontend can show transparency.
    cols = [
        "PLAY_TYPE",
        "PPP_PRED",
        "PPP_GAP",
        "PPP_OFF",
        "PPP_DEF",
        "PPP_OFF_SHRUNK",
        "PPP_DEF_SHRUNK",
        "PPP_LEAGUE_OFF",
        "PPP_LEAGUE_DEF",
        "RELIABILITY_WEIGHT_OFF",
        "RELIABILITY_WEIGHT_DEF",
        "REL_LEAGUE_OFF",
        "REL_LEAGUE_DEF",
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
        "SF_POSS_PCT_OFF",
        "SF_POSS_PCT_DEF",
        "FT_POSS_PCT_OFF",
        "FT_POSS_PCT_DEF",
        "PLUSONE_POSS_PCT_OFF",
        "PLUSONE_POSS_PCT_DEF",
        "RATIONALE",
    ]

    top = merged.head(k)[cols].reset_index(drop=True)

    # Store normalized weights as columns so the UI can display them.
    top["W_OFF"] = w_off
    top["W_DEF"] = w_def

    return top


class BaselineRecommender:
    """Load the Synergy snapshot once and expose baseline utilities."""

    def __init__(self, synergy_csv_path: str):
        synergy_csv_path = Path(synergy_csv_path)
        if not synergy_csv_path.exists():
            raise FileNotFoundError(synergy_csv_path)

        self.synergy_csv_path = synergy_csv_path

        # Read the snapshot.
        self.raw_df = pd.read_csv(synergy_csv_path)

        # Build baseline tables (team + league) once at startup.
        self.team_df, self.league_df = prepare_baseline_tables(self.raw_df)

        # Convenience views.
        self.offense_tbl = self.team_df[self.team_df["SIDE"] == "offense"].copy()
        self.defense_tbl = self.team_df[self.team_df["SIDE"] == "defense"].copy()

    def rank(
        self,
        season: str,
        our_team: str,
        opp_team: str,
        k: int = 5,
        w_off: float = 0.7,
        w_def: float = 0.3,
    ) -> pd.DataFrame:
        """Convenience wrapper used by the FastAPI layer."""

        return rank_playtypes_baseline(
            self.team_df,
            self.league_df,
            season=season,
            our_team=our_team,
            opp_team=opp_team,
            k=k,
            w_off=w_off,
            w_def=w_def,
        )


if __name__ == "__main__":
    # Manual smoke test.
    csv_path = Path(__file__).parent / "data" / "synergy_playtypes_2019_2025_players.csv"
    rec = BaselineRecommender(str(csv_path))
    print(rec.rank("2019-20", "LAL", "BOS", k=5))
