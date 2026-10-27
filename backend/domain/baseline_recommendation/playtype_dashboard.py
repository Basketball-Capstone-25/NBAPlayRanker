"""Historical play-type profiles for the coach matchup dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd

from domain.baseline_recommendation.baseline_recommender import AGGREGATION_VERSION, SEASON_PHASE


class PlaytypeDashboardService:
    def __init__(self, team_df: pd.DataFrame, league_df: pd.DataFrame, dataset_version: str):
        self.team_df = team_df
        self.league_df = league_df
        self.dataset_version = dataset_version

    def build(self, season: str, our_team: str, opp_team: str, min_poss: int = 20) -> dict[str, Any]:
        if isinstance(min_poss, bool) or not isinstance(min_poss, int) or not 1 <= min_poss <= 10000:
            raise ValueError("Minimum possessions must be an integer between 1 and 10000.")
        season_rows = self.team_df.loc[self.team_df["SEASON"].eq(season)]
        if season_rows.empty:
            raise ValueError("Select an available season.")
        teams = set(season_rows["TEAM_ABBREVIATION"])
        if our_team not in teams or opp_team not in teams:
            raise ValueError("Select teams with data in the selected season.")
        if our_team == opp_team:
            raise ValueError("Our team and opponent must be different.")
        offense = season_rows.loc[season_rows["SIDE"].eq("offense")]
        our = offense.loc[offense["TEAM_ABBREVIATION"].eq(our_team)]
        opponent = season_rows.loc[
            season_rows["SIDE"].eq("defense") & season_rows["TEAM_ABBREVIATION"].eq(opp_team)
        ]
        league = self.league_df.loc[self.league_df["SEASON"].eq(season)]
        if our.empty or opponent.empty:
            raise ValueError("Offense or defensive profile data is unavailable for this matchup.")
        unavailable_defense = sorted(set(offense["PLAY_TYPE"]) - set(opponent["PLAY_TYPE"]))
        warnings = ["These are historical regular-season profiles, not a forecast for this game."]
        if unavailable_defense:
            warnings.append(
                f"Opponent defensive data covers {len(opponent)} play types; missing categories are unavailable, not zero."
            )
        warnings.append("The sample cutoff is a display rule, not a confidence interval.")
        return {
            "schema_version": "playtype_dashboard_v1",
            "aggregation_version": AGGREGATION_VERSION,
            "filters": {"season": season, "season_phase": SEASON_PHASE, "our": our_team, "opp": opp_team, "min_poss": min_poss},
            "dataset_versions": {"synergy": self.dataset_version},
            "coverage": {
                "offense_types": int(len(our)),
                "defense_types": int(len(opponent)),
                "unavailable_defense_types": unavailable_defense,
            },
            "usage_efficiency": self._usage_efficiency(our, league.loc[league["SIDE"].eq("offense")], our_team, min_poss),
            "opponent_defense": self._opponent_defense(opponent, league.loc[league["SIDE"].eq("defense")], opp_team, min_poss),
            "warnings": warnings,
        }

    def _usage_efficiency(self, our: pd.DataFrame, league: pd.DataFrame, team: str, min_poss: int) -> dict[str, Any]:
        total = float(our["POSS"].sum())
        league_total = float(league["LEAGUE_POSS"].sum())
        reference = league.set_index("PLAY_TYPE")
        rows = []
        for _, row in our.sort_values("PLAY_TYPE").iterrows():
            benchmark = reference.loc[row["PLAY_TYPE"]]
            usage = 100.0 * float(row["POSS"]) / total
            league_usage = 100.0 * float(benchmark["LEAGUE_POSS"]) / league_total
            rows.append({
                "play_type": str(row["PLAY_TYPE"]), "possessions": int(row["POSS"]), "points": int(row["PTS"]),
                "usage_pct": usage, "ppp": float(row["PPP"]),
                "league_usage_pct": league_usage, "league_ppp": float(benchmark["PPP"]),
                "usage_delta_pp": usage - league_usage, "ppp_delta": float(row["PPP"] - benchmark["PPP"]),
                "sample_ok": bool(row["POSS"] >= min_poss),
            })
        return {"team": team, "total_possessions": int(total), "league_total_possessions": int(league_total), "rows": rows}

    def _opponent_defense(self, opponent: pd.DataFrame, league: pd.DataFrame, team: str, min_poss: int) -> dict[str, Any]:
        reference = league.set_index("PLAY_TYPE")
        rows = []
        for _, row in opponent.iterrows():
            benchmark = reference.loc[row["PLAY_TYPE"]]
            rows.append({
                "play_type": str(row["PLAY_TYPE"]), "possessions": int(row["POSS"]), "points_allowed": int(row["PTS"]),
                "allowed_ppp": float(row["PPP"]), "league_allowed_ppp": float(benchmark["PPP"]),
                "allowed_ppp_delta": float(row["PPP"] - benchmark["PPP"]),
                "sample_ok": bool(row["POSS"] >= min_poss), "weak_spot_rank": None,
            })
        rows.sort(key=lambda row: (-row["allowed_ppp_delta"], -row["possessions"], row["play_type"]))
        candidates = [row for row in rows if row["sample_ok"] and row["allowed_ppp_delta"] > 1e-12]
        for rank, row in enumerate(candidates[:3], 1):
            row["weak_spot_rank"] = rank
        return {"team": team, "tracked_possessions": int(opponent["POSS"].sum()), "rows": rows}
