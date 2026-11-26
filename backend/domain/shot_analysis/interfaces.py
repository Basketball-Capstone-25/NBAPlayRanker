"""Shot analysis interfaces – public contracts for the Shot Analysis subsystem."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple

import pandas as pd


class IRunShotModelCV(Protocol):
    """Cross-validate shot-level regressors and return summary + fold data."""

    def __call__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        max_rows: Optional[int] = 75_000,
        include_shooter: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return model summaries and per-fold results for a game-grouped shot evaluation."""
        ...


class IComputeShotMLAnalysis(Protocol):
    """Run the full exploratory shot-ML analysis pipeline."""

    def __call__(
        self,
        *,
        n_splits: int = 5,
        max_rows: int = 200_000,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Build or retrieve the analysis for the requested sample limit and fold count."""
        ...


class IBuildShotAggregates(Protocol):
    """Aggregate shot stats by type, zone, and type-zone from cleaned data."""

    def __call__(self, clean_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize cleaned attempts by team for both shots taken and shots allowed."""
        ...


class IBuildLeagueBaselines(Protocol):
    """Compute league-average baselines from aggregated shot data."""

    def __call__(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """Combine team shot totals into league reference rates for each category."""
        ...


class IBuildShotsClean(Protocol):
    """Clean and standardize raw PBP shot data into canonical format."""

    def __call__(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw event rows to the shot schema consumed by the analysis and ranking code."""
        ...


class IGetShotsCleanDf(Protocol):
    """Load the cleaned shots DataFrame (process-level cached)."""

    def __call__(self, *, force_reload: bool = False) -> pd.DataFrame:
        """Supply the cached shot table or reread it when a reload is requested."""
        ...
