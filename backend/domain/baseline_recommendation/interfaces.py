"""Baseline recommendation interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd

class IBaselineRecommender(ABC):
    """Interface for the baseline play-type ranker."""

    @abstractmethod
    def rank(
        self,
        season: str,
        our_team: str,
        opp_team: str,
        k: int = 5,
    ) -> pd.DataFrame:
        """Return top-k ranked play types for a matchup."""
        ...

class IShotBaselineRecommender(ABC):
    """Interface for the shot-type/zone ranker (Dataset2)."""

    @abstractmethod
    def rank(
        self,
        season: str,
        our_team: str,
        opp_team: str,
        k: int = 5,
        w_off: float = 0.7,
    ) -> Dict[str, Any]:
        """Return top-k ranked shot types/zones for a matchup."""
        ...
