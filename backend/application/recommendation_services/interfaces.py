"""Recommendation service interfaces."""
from __future__ import annotations

from domain.baseline_recommendation import BaselineRecommender

from typing import Protocol

import pandas as pd

class IGetBaselineRecommendations(Protocol):
    """Protocol for orchestrating a baseline play-type ranking."""

    def __call__(
        self,
        rec: BaselineRecommender,
        season: str,
        our_team: str,
        opp_team: str,
        k: int = 5,
        w_off: float = 0.7,
        w_def: float = 0.3,
    ) -> pd.DataFrame: ...

class IGetContextMLRecommendations(Protocol):
    """Protocol for orchestrating a context-ML ranking request."""

    def __call__(
        self,
        season: str,
        our_team: str,
        opp_team: str,
        margin: float,
        period: int,
        time_remaining_period_sec: float,
        k: int = 5,
        w_off: float = 0.7,
        w_def: float = 0.3,
    ) -> pd.DataFrame: ...
