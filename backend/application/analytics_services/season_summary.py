"""A compact season summary using the existing historical Top-K calculation."""

import math
import re

from domain.baseline_recommendation import BaselineRecommender, rank_playtypes_baseline
from domain.baseline_recommendation.baseline_recommender import AGGREGATION_VERSION
from domain.statistical_analysis.topk_uplift import compute_topk_uplift


def build_season_summary(
    rec: BaselineRecommender, season: str, our: str, opp: str, k: int = 5,
    w_off: float = 0.7,
) -> dict:
    if not re.fullmatch(r"20\d{2}-\d{2}", season) or season[-2:] != str(int(season[:4]) + 1)[-2:]:
        raise ValueError("Use a season such as 2024-25.")
    teams = set(rec.team_df["TEAM_ABBREVIATION"])
    if our not in teams or opp not in teams:
        raise ValueError("Select two known teams.")
    if our == opp:
        raise ValueError("Our team and opponent must be different.")
    if not 1 <= k <= 10:
        raise ValueError("k must be between 1 and 10.")
    if not math.isfinite(w_off) or not 0 <= w_off <= 1:
        raise ValueError("Offense weight must be between 0 and 1.")

    result = {
        "status": "no_data",
        "message": "No data for this period.",
        "filters": {"period": "season", "season": season, "our": our, "opp": opp,
                    "k": k, "w_off": w_off, "w_def": 1 - w_off,
                    "side": "offense", "season_phase": "regular_season"},
        "dataset_version": rec.raw_df.attrs.get("dataset_version", "unavailable"),
        "aggregation_version": AGGREGATION_VERSION,
        "summary": None,
        "play_types": [],
        "limitations": [
            "Season totals only; this source cannot provide last-5 or last-10-game comparisons.",
            "Modeled uplift is a historical comparison, not measured improvement or a future forecast.",
            "Usage covers recorded offensive play-type possessions, not independently verified whole-game possessions.",
        ],
    }
    frame = rec.team_df.loc[rec.team_df["SEASON"].eq(season)]
    offense = frame.loc[frame["TEAM_ABBREVIATION"].eq(our) & frame["SIDE"].eq("offense")]
    defense = frame.loc[frame["TEAM_ABBREVIATION"].eq(opp) & frame["SIDE"].eq("defense")]
    if not set(offense["PLAY_TYPE"]) & set(defense["PLAY_TYPE"]):
        return result

    rankings = rank_playtypes_baseline(
        rec.team_df, rec.league_df, season, our, opp, k=k, w_off=w_off, w_def=1 - w_off,
    )
    evidence = compute_topk_uplift(
        rec.team_df, rankings, season=season, our_team=our, opp_team=opp,
        k=k, w_off=w_off, provenance={"dataset_version": result["dataset_version"]},
    )
    total = evidence["team_season"]["possessions_denominator"]
    selected = evidence["topk"]["possessions_denominator"]
    result.update({
        "status": "ok",
        "message": None,
        "summary": {
            "observed_team_ppp": evidence["team_season"]["ppp"],
            "modeled_topk_ppp": evidence["topk"]["modeled_ppp"],
            "modeled_uplift_ppp": evidence["uplift_ppp"],
            "team_possessions": total,
            "selected_possessions": selected,
            "selected_usage_pct": 100 * selected / total,
            "k_returned": evidence["k_returned"],
        },
        "play_types": [
            {"rank": row["rank"], "play_type": row["play_type"],
             "possessions": row["historical_offense_possessions"],
             # Use the full season, not the renormalized Top-K mixture.
             "usage_pct": 100 * row["historical_offense_possessions"] / total}
            for row in evidence["rankings"]
        ],
    })
    return result
