"""Describe modeled Top-K PPP relative to the observed team-season average.

This is an in-sample recommendation diagnostic, not a causal estimate or a
held-out assessment of what would happen if a team followed the recommendations.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


class UpliftDataUnavailable(ValueError):
    """The requested slice cannot support a defensible PPP comparison."""


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise UpliftDataUnavailable(f"Required source column '{column}' is missing.")
    values = pd.to_numeric(frame[column], errors="coerce")
    if not values.map(lambda value: math.isfinite(float(value))).all():
        raise UpliftDataUnavailable(f"Source column '{column}' contains non-finite values.")
    return values.astype(float)


def compute_topk_uplift(
    team_df: pd.DataFrame,
    rankings: pd.DataFrame,
    *,
    season: str,
    our_team: str,
    opp_team: str,
    k: int,
    w_off: float,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Use season-wide PTS/POSS and renormalized historical Top-K usage weights.

The denominator includes every available offensive play type for our team in
the season, including those absent from the opponent's matched ranking. PTS is
required: substituting rounded PPP * POSS would silently change the estimand.
"""
    if our_team == opp_team:
        raise ValueError("Our team and opponent must be different.")
    if not 1 <= k <= 10 or not math.isfinite(w_off) or not 0 <= w_off <= 1:
        raise ValueError("k must be 1..10 and w_off must be a finite value in [0, 1].")
    scope_columns = {"SEASON", "TEAM_ABBREVIATION", "SIDE"}
    if not scope_columns.issubset(team_df.columns):
        raise UpliftDataUnavailable("Team-season scope columns are missing.")
    season_offense = team_df.loc[
        (team_df["SEASON"] == season)
        & (team_df["TEAM_ABBREVIATION"] == our_team)
        & (team_df["SIDE"] == "offense")
    ]
    if season_offense.empty or rankings.empty:
        raise UpliftDataUnavailable("No offensive data or eligible recommendations for this matchup.")
    if len(rankings) > k:
        raise ValueError("Rankings contain more rows than the requested k.")
    if "PLAY_TYPE" not in rankings or rankings["PLAY_TYPE"].isna().any():
        raise UpliftDataUnavailable("Recommendation play types are missing.")
    if rankings["PLAY_TYPE"].duplicated().any():
        raise UpliftDataUnavailable("Recommendation play types must be unique.")

    points = _numeric(season_offense, "PTS")
    possessions = _numeric(season_offense, "POSS")
    selected_possessions = _numeric(rankings, "POSS_OFF")
    predictions = _numeric(rankings, "PPP_PRED")
    if (points < 0).any() or (possessions < 0).any() or (selected_possessions < 0).any():
        raise UpliftDataUnavailable("Source points and possessions must be non-negative.")
    if ((possessions == 0) & (points > 0)).any():
        raise UpliftDataUnavailable("Source contains points without any possessions.")
    total_points = float(points.sum())
    total_possessions = float(possessions.sum())
    topk_possessions = float(selected_possessions.sum())
    if total_possessions <= 0 or topk_possessions <= 0:
        raise UpliftDataUnavailable("Positive team-season and selected-play possessions are required.")

    modeled_numerator = float((predictions * selected_possessions).sum())
    baseline_ppp = total_points / total_possessions
    topk_ppp = modeled_numerator / topk_possessions
    uplift = topk_ppp - baseline_ppp
    rows = []
    for rank, (_, row) in enumerate(rankings.iterrows(), start=1):
        ppp = float(row["PPP_PRED"])
        poss = float(row["POSS_OFF"])
        rows.append({
            "rank": rank,
            "play_type": str(row["PLAY_TYPE"]),
            "modeled_ppp": ppp,
            "historical_offense_possessions": poss,
            "topk_weight": poss / topk_possessions,
            "modeled_points_contribution": ppp * poss,
            "uplift_vs_team_season_ppp": ppp - baseline_ppp,
        })
    return {
        "metric": "topk_ppp_uplift",
        "metric_version": "1",
        "season": season,
        "our_team": our_team,
        "opp_team": opp_team,
        "k": k,
        "k_returned": len(rows),
        "filters": {"season": season, "our": our_team, "opp": opp_team, "k": k,
                    "w_off": w_off, "w_def": 1.0 - w_off, "side": "offense"},
        "units": "points per possession",
        "ranking_model": "existing possession-shrunk baseline matchup recommender",
        "estimand": "historical modeled recommendation comparison",
        "evaluation_scope": "in-sample descriptive diagnostic; not held-out evaluation",
        "team_season": {
            "ppp": baseline_ppp,
            "points_numerator": total_points,
            "possessions_denominator": total_possessions,
            "play_type_rows": len(season_offense),
            "scope": "all available offensive Synergy play-type rows for our team and season",
            "formula": "sum(PTS) / sum(POSS)",
        },
        "topk": {
            "modeled_ppp": topk_ppp,
            "modeled_points_numerator": modeled_numerator,
            "possessions_denominator": topk_possessions,
            "weighting": "historical offensive possessions, renormalized over selected play types",
            "formula": "sum(PPP_PRED * POSS_OFF) / sum(POSS_OFF)",
        },
        "uplift_ppp": uplift,
        "uplift_percent": 100.0 * uplift / baseline_ppp if baseline_ppp != 0 else None,
        "relative_uplift_unavailable_reason": (
            "Team seasonal PPP is zero; relative uplift is undefined." if baseline_ppp == 0 else None
        ),
        "uplift_formula": "topk.modeled_ppp - team_season.ppp",
        "relative_uplift_formula": "100 * uplift_ppp / team_season.ppp",
        "rankings": rows,
        "data_provenance": provenance,
        "limitations": [
            "The baseline covers available Synergy play-type possessions, not independently verified whole-game possessions.",
            "Recommendations and team-season reference use the same historical season; this is not out-of-sample evidence.",
            "The selected mixture retains historical usage proportions; it is not a proposed equal-usage policy.",
            "Modeled PPP and uplift do not demonstrate a causal improvement in future or real-game scoring.",
        ],
    }
