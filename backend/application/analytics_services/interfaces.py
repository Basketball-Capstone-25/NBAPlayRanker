"""Analytics service interfaces."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

class IGetCVEvaluation(Protocol):
    """Protocol for cross-validation evaluation."""

    def __call__(self) -> Dict[str, Any]:
        """Request the evaluation summary through the analytics service contract."""
        ...

class IGetPairedTTest(Protocol):
    """Protocol for paired t-test comparing model RMSEs."""

    def __call__(self) -> Tuple[Optional[float], Optional[float]]:
        """Return the paired statistic and p-value for fold-level errors when available."""
        ...

class IGetMLAnalysis(Protocol):
    """Protocol for full ML analysis for a given season."""

    def __call__(self, season: str) -> Dict[str, Any]:
        """Describe the season argument expected by the original play-type analysis service contract."""
        ...

class IGetShotMLAnalysis(Protocol):
    """Protocol for shot ML analysis for a given season."""

    def __call__(self, season: str) -> Dict[str, Any]:
        """Describe the season argument expected by the original shot-analysis service contract."""
        ...
