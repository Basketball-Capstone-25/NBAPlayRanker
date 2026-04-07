from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from domain.baseline_recommendation.interfaces import IShotBaselineRecommender
from domain.shot_analysis.shot_aggregates import AGG_PARQUET

LEAGUE_PARQUET = AGG_PARQUET.with_name("shots_agg_league.parquet")

def _require_in(value: str, allowed: List[str], label: str) -> None:
    if value not in allowed:
        raise ValueError(f"Unknown {label} '{value}'. Allowed: {allowed}")

def _df_to_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    clean = df.replace({np.nan: None})
    return clean.to_dict(orient="records")

def _shrink_eva(team_epa: pd.Series, league_epa: pd.Series, rel: pd.Series) -> pd.Series:
    return rel * team_epa + (1 - rel) * league_epa

def _rank_level(
    *,
    level: str,
    agg_df: pd.DataFrame,
    league_df: pd.DataFrame,
    season: str,
    our_team: str,
    opp_team: str,
    k: int,
    w_off: float,
    w_def: float,
) -> pd.DataFrame:
    level_df = agg_df[agg_df["LEVEL"] == level].copy()
    league_level = league_df[league_df["LEVEL"] == level].copy()

    off = level_df.query("SEASON_STR == @season and TEAM_ABBR == @our_team and ROLE == 'offense'").copy()
    deff = level_df.query("SEASON_STR == @season and TEAM_ABBR == @opp_team and ROLE == 'defense_allowed'").copy()

    if off.empty or deff.empty:
        raise ValueError("No offense/defense data for this matchup/season at this level.")

    keys = ["SHOT_TYPE"] if level == "shot_type" else ["ZONE"] if level == "zone" else ["SHOT_TYPE", "ZONE"]

    league_off = league_level.query("SEASON_STR == @season and ROLE == 'offense'")[keys + ["EPA_LEAGUE"]].rename(
        columns={"EPA_LEAGUE": "EPA_LEAGUE_OFF"}
    )
    league_def = league_level.query("SEASON_STR == @season and ROLE == 'defense_allowed'")[keys + ["EPA_LEAGUE"]].rename(
        columns={"EPA_LEAGUE": "EPA_LEAGUE_DEF"}
    )

    merged = off.merge(
        deff[keys + ["EPA", "attempts", "RELIABILITY_WEIGHT"]],
        on=keys,
        how="inner",
        suffixes=("_OFF", "_DEF"),
    )

    merged = merged.merge(league_off, on=keys, how="left")
    merged = merged.merge(league_def, on=keys, how="left")

    rel_off = merged["RELIABILITY_WEIGHT_OFF"]
    rel_def = merged["RELIABILITY_WEIGHT_DEF"]

    merged["EPA_OFF_SHRUNK"] = _shrink_eva(merged["EPA_OFF"], merged["EPA_LEAGUE_OFF"], rel_off)
    merged["EPA_DEF_SHRUNK"] = _shrink_eva(merged["EPA_DEF"], merged["EPA_LEAGUE_DEF"], rel_def)

    merged["EPA_PRED"] = float(w_off) * merged["EPA_OFF_SHRUNK"] + float(w_def) * merged["EPA_DEF_SHRUNK"]

    def _rationale(row: pd.Series) -> str:
        return (
            f"EPA {row['EPA_PRED']:.3f} = "
            f"{w_off:.2f}*{row['EPA_OFF_SHRUNK']:.3f} + {w_def:.2f}*{row['EPA_DEF_SHRUNK']:.3f} "
            f"(rel off={row['RELIABILITY_WEIGHT_OFF']:.2f}, def={row['RELIABILITY_WEIGHT_DEF']:.2f})."
        )

    merged["RATIONALE"] = merged.apply(_rationale, axis=1)

    merged = merged.sort_values(["EPA_PRED", "attempts_OFF"], ascending=False).head(k).reset_index(drop=True)

    cols = keys + [
        "EPA_PRED",
        "EPA_OFF_SHRUNK",
        "EPA_DEF_SHRUNK",
        "EPA_OFF",
        "EPA_DEF",
        "EPA_LEAGUE_OFF",
        "EPA_LEAGUE_DEF",
        "attempts_OFF",
        "attempts_DEF",
        "RELIABILITY_WEIGHT_OFF",
        "RELIABILITY_WEIGHT_DEF",
        "RATIONALE",
    ]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols].copy()

def rank_shot_plan_baseline(
    *,
    agg_df: pd.DataFrame,
    league_df: pd.DataFrame,
    season: str,
    our_team: str,
    opp_team: str,
    k: int = 5,
    w_off: float = 0.7,
    w_def: float = 0.3,
) -> Dict[str, List[Dict[str, object]]]:
    valid_seasons = sorted(agg_df["SEASON_STR"].dropna().unique().tolist())
    valid_teams = sorted(agg_df["TEAM_ABBR"].dropna().unique().tolist())

    _require_in(season, valid_seasons, "season")
    _require_in(our_team, valid_teams, "our team")
    _require_in(opp_team, valid_teams, "opponent team")
    if our_team == opp_team:
        raise ValueError("Our team and opponent must be different.")
    if not (1 <= k <= 10):
        raise ValueError("k must be between 1 and 10.")

    top_shot_types = _rank_level(
        level="shot_type",
        agg_df=agg_df,
        league_df=league_df,
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        k=k,
        w_off=w_off,
        w_def=w_def,
    )
    top_zones = _rank_level(
        level="zone",
        agg_df=agg_df,
        league_df=league_df,
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        k=k,
        w_off=w_off,
        w_def=w_def,
    )

    return {
        "top_shot_types": _df_to_records(top_shot_types),
        "top_zones": _df_to_records(top_zones),
    }

class ShotBaselineRecommender(IShotBaselineRecommender):
    """Loads shots_agg.parquet + shots_agg_league.parquet once, provides ranking."""

    def __init__(self, agg_path: Path = AGG_PARQUET, league_path: Path = LEAGUE_PARQUET):
        agg_path = Path(agg_path)
        league_path = Path(league_path)
        if not agg_path.exists():
            raise FileNotFoundError(agg_path)
        if not league_path.exists():
            raise FileNotFoundError(league_path)

        self.agg_df = pd.read_parquet(agg_path)
        self.league_df = pd.read_parquet(league_path)
        # ADDED: lightweight meta caches for dropdowns (no parquet scans later)
        # Used by /pbp/meta/options (and any other UI dropdown needs)
        self.available_seasons: List[str] = (
            sorted(self.agg_df["SEASON_STR"].dropna().astype(str).unique().tolist())
            if "SEASON_STR" in self.agg_df.columns
            else []
        )
        self.available_teams: List[str] = (
            sorted(self.agg_df["TEAM_ABBR"].dropna().astype(str).unique().tolist())
            if "TEAM_ABBR" in self.agg_df.columns
            else []
        )
        self.shot_types: List[str] = (
            sorted(
                self.agg_df.loc[self.agg_df["LEVEL"] == "shot_type", "SHOT_TYPE"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if ("LEVEL" in self.agg_df.columns and "SHOT_TYPE" in self.agg_df.columns)
            else []
        )
        self.zones: List[str] = (
            sorted(
                self.agg_df.loc[self.agg_df["LEVEL"] == "zone", "ZONE"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if ("LEVEL" in self.agg_df.columns and "ZONE" in self.agg_df.columns)
            else []
        )

    def rank(
        self,
        season: str,
        our_team: str,
        opp_team: str,
        k: int = 5,
        w_off: float = 0.7,
    ) -> Dict[str, List[Dict[str, object]]]:
        w_def = float(1.0 - w_off)
        return rank_shot_plan_baseline(
            agg_df=self.agg_df,
            league_df=self.league_df,
            season=season,
            our_team=our_team,
            opp_team=opp_team,
            k=k,
            w_off=float(w_off),
            w_def=w_def,
        )
