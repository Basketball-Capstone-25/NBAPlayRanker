"""Statistical analysis interfaces."""
from __future__ import annotations

from typing import Any, Dict, List, Protocol

import pandas as pd

class IComputeMLAnalysis(Protocol):
    """Protocol for the full ML statistical-analysis pipeline."""

    def __call__(
        self,
        team_df: pd.DataFrame,
        league_df: pd.DataFrame,
        *,
        n_splits: int = 5,
        min_poss: int = 25,
        force_refresh: bool = False,
    ) -> Dict[str, Any]: ...
