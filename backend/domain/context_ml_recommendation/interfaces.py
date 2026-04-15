"""Context ML recommendation interfaces."""
from __future__ import annotations

from typing import Protocol, Tuple

import pandas as pd

# Functional protocols – each describes a callable with a matching signature

class IRankMLWithContext(Protocol):
    """Protocol for the main context-aware ML ranking entry-point."""

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

class IComputeContextFactors(Protocol):
    """Protocol for computing late-game / trailing / leading factors."""

    def __call__(
        self,
        margin: float,
        period: int,
        time_remaining_period_sec: float,
    ) -> Tuple[float, float, float]: ...

class IBuildMLMatchupTable(Protocol):
    """Protocol for building the ML-enhanced matchup table."""

    def __call__(
        self,
        season: str,
        our_team: str,
        opp_team: str,
        w_off: float = 0.7,
        w_def: float = 0.3,
    ) -> pd.DataFrame: ...

class IApplyContextAdjustments(Protocol):
    """Protocol for applying context-based score adjustments."""

    def __call__(
        self,
        df: pd.DataFrame,
        margin: float,
        period: int,
        time_remaining_period_sec: float,
    ) -> pd.DataFrame: ...
