"""Recommendation service: baseline and context-ML ranking."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from domain.baseline_recommendation import BaselineRecommender, rank_playtypes_baseline
from domain.context_ml_recommendation import rank_ml_with_context


def get_baseline_recommendations(
    rec: BaselineRecommender,
    season: str,
    our_team: str,
    opp_team: str,
    k: int = 5,
    w_off: float = 0.7,
    w_def: float = 0.3,
) -> pd.DataFrame:
    """Orchestrate a baseline play-type ranking."""
    return rank_playtypes_baseline(
        team_df=rec.team_df,
        league_df=rec.league_df,
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        k=k,
        w_off=w_off,
        w_def=w_def,
    )


def get_context_ml_recommendations(
    season: str,
    our_team: str,
    opp_team: str,
    margin: float,
    period: int,
    time_remaining_period_sec: float,
    k: int = 5,
    w_off: float = 0.7,
    w_def: float = 0.3,
) -> pd.DataFrame:
    """Orchestrate a context-aware ML-based play-type ranking."""
    return rank_ml_with_context(
        season=season,
        our_team=our_team,
        opp_team=opp_team,
        margin=margin,
        period=period,
        time_remaining_period_sec=time_remaining_period_sec,
        k=k,
        w_off=w_off,
        w_def=w_def,
    )